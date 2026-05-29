import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "False"

import argparse
import numpy as np
from pymbar import MBAR


kB = 0.0083144621  # kJ/mol/K


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compute unbiased MBAR weights from umbrella sampling data."
    )

    parser.add_argument(
        "-f",
        "--input_file",
        required=True,
        type=str,
        help="Input metadata file containing paths, centers, force constants, and temperatures.",
    )

    parser.add_argument(
        "-target_temp",
        "--target_temp",
        required=True,
        type=float,
        help="Target temperature for reweighting in Kelvin.",
    )

    parser.add_argument(
        "-stride",
        "--stride",
        required=True,
        type=int,
        help="Read every nth data point from each trajectory.",
    )

    parser.add_argument(
        "-N_CV",
        "--N_CV",
        required=True,
        type=int,
        help="Number of biased collective variables.",
    )

    return parser.parse_args()


def read_metadata(input_file, D):
    paths = []
    centers = []
    ksprings = []
    temps = []

    with open(input_file) as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            tokens = line.split()

            expected_min_columns = 1 + 2 * D + 1
            if len(tokens) < expected_min_columns:
                raise ValueError(
                    f"Line {line_number} in {input_file} has too few columns.\n"
                    f"Expected at least {expected_min_columns} columns for D={D}, "
                    f"but got {len(tokens)}."
                )

            path = tokens[0]
            c = list(map(float, tokens[1:1 + D]))
            k = list(map(float, tokens[1 + D:1 + 2 * D]))
            T = float(tokens[-1])

            paths.append(path)
            centers.append(c)
            ksprings.append(k)
            temps.append(T)

    if len(paths) == 0:
        raise ValueError(f"No umbrella windows found in {input_file}.")

    return (
        np.array(paths),
        np.array(centers, dtype=np.float64),
        np.array(ksprings, dtype=np.float64),
        np.array(temps, dtype=np.float64),
    )


def load_window_data(path, D, stride):
    data = np.loadtxt(path)
    data = np.atleast_2d(data)

    required_columns = 1 + D + 1
    if data.shape[1] < required_columns:
        raise ValueError(
            f"File {path} has too few columns.\n"
            f"Expected at least {required_columns} columns: "
            f"time + {D} CVs + potential energy."
        )

    data = data[::stride]

    time = data[:, 0]
    CV = data[:, 1:1 + D]
    U = data[:, 1 + D]

    return time, CV, U


def main():
    args = parse_arguments()

    input_file = args.input_file
    target_temp = args.target_temp
    stride = args.stride
    D = args.N_CV

    if D <= 0:
        raise ValueError("-N_CV must be a positive integer.")

    if stride <= 0:
        raise ValueError("-stride must be a positive integer.")

    print("\n========== MBAR WEIGHT CALCULATION ==========")
    print(f"Input file       : {input_file}")
    print(f"Target temp      : {target_temp:.6f} K")
    print(f"Stride           : {stride}")
    print(f"Number of CVs    : {D}")

    # ---------- STEP 1: READ INPUT FILE ----------
    paths, centers, ksprings, temps = read_metadata(input_file, D)
    K = len(paths)

    print(f"\nDetected {K} umbrella windows with {D} CVs.")

    # ---------- STEP 2: LOAD DATA ----------
    time_all_by_window = []
    cv_all = []
    U_all = []
    N_k = []

    print("\nLoading trajectory data...")

    for k in range(K):
        time_k, CV_k, U_k = load_window_data(paths[k], D, stride)

        time_all_by_window.append(time_k)
        cv_all.append(CV_k)
        U_all.append(U_k)
        N_k.append(len(U_k))

        print(
            f"Window {k + 1:4d}: {len(U_k):8d} samples "
            f"from {paths[k]}"
        )

    cv_all = np.concatenate(cv_all, axis=0)
    U_all = np.concatenate(U_all, axis=0)
    N_k = np.array(N_k, dtype=int)

    N = len(U_all)

    print(f"\nTotal samples: {N}")

    if np.any(N_k == 0):
        empty_windows = np.where(N_k == 0)[0] + 1
        raise ValueError(f"Empty windows after applying stride: {empty_windows}")

    # ---------- STEP 3: BUILD u_kn ----------
    print("\nBuilding reduced potential matrix u_kn...")

    u_kn = np.zeros((K, N), dtype=np.float32)

    L = 2.0 * np.pi
    periodic = np.array([True] * D)

    for k in range(K):
        beta_k = 1.0 / (kB * temps[k])

        diff = cv_all - centers[k]

        for d in range(D):
            if periodic[d]:
                diff[:, d] -= L * np.round(diff[:, d] / L)

        bias = 0.5 * np.sum(ksprings[k] * diff**2, axis=1)

        u_kn[k, :] = beta_k * (U_all + bias)

    # ---------- STEP 4: RUN PYMBAR ----------
    print("\nRunning PyMBAR...")

    mbar = MBAR(u_kn, N_k, verbose=True)
    f_k = mbar.f_k

    # ---------- STEP 5: COMPUTE UNBIASED WEIGHTS ----------
    print("\nComputing unbiased weights...")

    beta0 = 1.0 / (kB * target_temp)
    u0_n = beta0 * U_all

    # denominator: sum_k N_k * exp(f_k - u_kn)
    log_terms = f_k[:, None] - u_kn + np.log(N_k[:, None])

    # stable log-sum-exp
    max_log = np.max(log_terms, axis=0)
    log_denom = max_log + np.log(np.sum(np.exp(log_terms - max_log), axis=0))

    log_w_n = -u0_n - log_denom
    w_n = np.exp(log_w_n)

    w_n /= np.sum(w_n)

    print(f"Weight normalization check: {np.sum(w_n):.12f}")

    # ---------- STEP 6: SPLIT WEIGHTS AND SAVE PER-WINDOW ----------
    output_base = "./output"
    os.makedirs(output_base, exist_ok=True)

    starts = np.concatenate([[0], np.cumsum(N_k[:-1])])

    cv_header = "  ".join([f"CV{d + 1}" for d in range(D)])
    col_header = f"time  {cv_header}  weight"

    print("\nSaving per-window weights...")

    for k in range(K):
        start = starts[k]
        end = start + N_k[k]

        w_k = w_n[start:end]
        cv_k = cv_all[start:end]
        time_k = time_all_by_window[k]

        out_array = np.column_stack([time_k, cv_k, w_k])

        folder = os.path.join(output_base, str(k + 1))
        os.makedirs(folder, exist_ok=True)

        out_path = os.path.join(folder, "weights.dat")

        np.savetxt(
            out_path,
            out_array,
            header=col_header,
            fmt=["%.6f"] + ["%.6f"] * D + ["%.10e"],
        )

        print(
            f"Window {k + 1:4d}: {N_k[k]:8d} samples -> {out_path}"
        )

    # ---------- STEP 7: SAVE GLOBAL FREE ENERGIES ----------
    fk_path = os.path.join(output_base, "fk.dat")

    np.savetxt(
        fk_path,
        f_k,
        header="dimensionless free energies",
        fmt="%.12e",
    )

    print(f"\nSaved MBAR free energies: {fk_path}")
    print("\nDone. Weights saved to per-window folders.")
    print("============================================\n")


if __name__ == "__main__":
    main()
