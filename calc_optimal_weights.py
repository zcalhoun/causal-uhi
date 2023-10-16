import os
import numpy as np
import argparse
from itertools import product
import csv
import logging

# SKLEARN modules
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.gaussian_process import GaussianProcessRegressor, kernels

from scipy.optimize import minimize
import pdb

from src.utils import (
    load_dataset,
    create_folds,
    standardize_data,
    generate_data,
    create_weight_matrix,
)


def main(args):
    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Initialize a logger
    logging.basicConfig(
        filename=os.path.join(args.output_dir, "output.log"),
        filemode="w",
        level=args.log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info(args)

    # Initialize the length scale values to consider
    # NLCD_LS = range(args.nlcd_ls_vals[0], args.nlcd_ls_vals[1])
    NDVI_LS = range(args.ndvi_ls_vals[0], args.ndvi_ls_vals[1])
    ALBEDO_LS = range(args.albedo_ls_vals[0], args.albedo_ls_vals[1])

    # Read in the data
    data = load_dataset(args.data_dir, args.window_size)

    # Create K-folds of the data
    folds = create_folds(data, args.k_folds * args.k_folds_size)

    # Create the headers to use for the CSV output file
    headers = [
        "fold",
        "iter",
        "ndvi_ls",
        "albedo_ls",
        "train_ridge_score",
        "train_gp_score",
        "train_score",
        "val_ridge_score",
        "val_gp_score",
        "val_score",
    ]

    # Create headings for coefs
    for i in range(19):
        headers.append(f"beta_{i}")

    headers.append("intercept")
    headers.extend(["matern_const", "matern_ls", "dp_const", "dot_prod_sigma"])

    # Create header in the cSV
    with open(os.path.join(args.output_dir, "results.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

    # For each of the fold
    for fold in range(0, args.k_folds):
        # Split into a training and validation set
        low_bound = fold * args.k_folds_size
        upper_bound = (fold + 1) * args.k_folds_size
        train_idx = np.where((folds < low_bound) | (folds >= upper_bound))[0]
        val_idx = np.where((folds >= low_bound) & (folds < upper_bound))[0]
        # val_idx = np.where(folds == fold)[0]

        # Record the fold used
        logging.info(f"Starting fold {fold}.")

        # For each of the possible combinations of weights
        for ndvi_ls, albedo_ls in product(NDVI_LS, ALBEDO_LS):
            logging.debug(
                f"Starting fold {fold} with ndvi_ls={ndvi_ls}, albedo_ls={albedo_ls}"
            )

            # Create the datasets using the given weights
            logging.debug("Creating datasets.")
            X_train, y_train = generate_data(
                data, train_idx, ndvi_ls, albedo_ls, args.window_size, args.use_coords
            )
            X_val, y_val = generate_data(
                data, val_idx, ndvi_ls, albedo_ls, args.window_size, args.use_coords
            )

            logging.debug("Standardizing datasets")
            # Standardize the data
            X_train, X_val, y_train, y_val, _, _, _, _ = standardize_data(
                X_train, X_val, y_train, y_val, scaler=args.scaler
            )

            logging.debug("Fitting model.")

            ridge_output_train = y_train
            for i in range(args.max_iter):
                row = [fold, i, ndvi_ls, albedo_ls]
                # Fit the model to the training set
                lm = Ridge(alpha=args.l2_alpha, fit_intercept=False)
                lm.fit(X_train, ridge_output_train)

                logging.debug("Calculating R2 scores.")
                # Calculate the train and validation score
                train_ridge_score = lm.score(X_train, y_train)
                val_ridge_score = lm.score(X_val, y_val)

                train_preds = lm.predict(X_train)
                val_preds = lm.predict(X_val)

                train_residuals = y_train - train_preds
                val_residuals = y_val - val_preds
                sample_points, sample_residuals = generate_samples(
                    data.coords[train_idx], train_residuals, args.n_samples
                )

                gp = fit_gp(
                    sample_points,
                    sample_residuals,
                    args.gp_constant_1,
                    args.gp_length_scale,
                    args.gp_constant_2,
                    args.gp_sigma_0,
                    args.gp_noise,
                )

                U_train = gp.predict(data.coords[train_idx])
                U_val = gp.predict(data.coords[val_idx])

                row.extend(
                    [
                        train_ridge_score,
                        r2_score(train_residuals, U_train),
                        r2_score(y_train, train_preds + U_train),
                        val_ridge_score,
                        r2_score(val_residuals, U_val),
                        r2_score(y_val, val_preds + U_val),
                    ]
                )

                # Add coefficients
                row.extend(lm.coef_)
                # Add intercept
                row.append(lm.intercept_)
                # Add GP parameters
                kp = gp.kernel_.get_params()
                # pdb.set_trace()
                row.extend(
                    [
                        np.sqrt(kp["k1__k1__constant_value"]),
                        kp["k1__k2__length_scale"],
                        np.sqrt(kp["k2__k1__constant_value"]),
                        kp["k2__k2__sigma_0"],
                    ]
                )

                # Report the validation set error along with the parameters and the fold value
                logging.debug("Writing row to CSV.")

                # Write the row to the CSV file
                with open(os.path.join(args.output_dir, "results.csv"), "a") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

                # Update the residuals
                ridge_output_train = y_train - U_train


def fit_gp(points, residuals, constant_1, length_scale, constant_2, sigma_0, noise):
    kernel = kernels.ConstantKernel(
        constant_1, constant_value_bounds="fixed"
    ) * kernels.Matern(
        length_scale=length_scale, nu=0.5, length_scale_bounds="fixed"
    ) + kernels.ConstantKernel(
        constant_2, constant_value_bounds="fixed"
    ) * kernels.DotProduct(
        sigma_0, sigma_0_bounds="fixed"
    )

    gpr = GaussianProcessRegressor(kernel=kernel, optimizer=optimizer, alpha=noise)

    gpr.fit(points, residuals)

    return gpr


def optimizer(obj_func, initial_theta, bounds):
    opt_res = minimize(
        obj_func,
        initial_theta,
        method="L-BFGS-B",
        bounds=bounds,
        jac=True,
        options={"maxiter": 1000},
    )
    return opt_res.x, opt_res.fun


def generate_samples(coords, residuals, N):
    sample_idx = np.random.choice(np.arange(coords.shape[0]), N, replace=False)
    sample_points = coords[sample_idx, :]
    sample_residuals = residuals[sample_idx]

    return sample_points, sample_residuals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate the optimal weights for the model"
    )

    # Create data arguments
    parser.add_argument("--data_dir", type=str, help="Path to the data directory")
    parser.add_argument(
        "--window_size", type=int, default=20, help="Size of the window "
    )
    parser.add_argument("--use_coords", action="store_true")

    # Create the model arguments
    parser.add_argument(
        "--l2_alpha", type=float, default=1e1, help="L2 regularization parameter"
    )

    # Create the training arguments
    parser.add_argument(
        "--k_folds",
        type=int,
        default=5,
        help="Number of folds to use for cross validation",
    )
    parser.add_argument(
        "--k_folds_size", type=int, default=1, help="How big the k-folds should be."
    )

    parser.add_argument(
        "--scaler", choices=["minmax", "standard"], default="standard", help="Scaler"
    )

    parser.add_argument("--max_iter", type=int, default=20, help="Max iterations")

    # Add the start and end ints for the NDVI length scale range
    parser.add_argument(
        "--ndvi_ls_vals",
        nargs="+",
        type=int,
        help="The start and end values for the NDVI length scale range",
        default=[1, 10],
    )

    # Add the start and end ints for the albedo length scale range
    parser.add_argument(
        "--albedo_ls_vals",
        nargs="+",
        type=int,
        help="The start and end values for the albedo length scale range",
        default=[1, 10],
    )

    # Add the GP arguments
    parser.add_argument(
        "--gp_constant_1",
        type=float,
        default=0.5,
        help="The constant parameter for the Gaussian Process.",
    )
    parser.add_argument(
        "--gp_constant_2",
        type=float,
        default=0.001,
        help="The constant parameter for the Gaussian Process.",
    )
    # Add a term for the noise
    parser.add_argument(
        "--gp_noise",
        type=float,
        default=0.1,
        help="The noise parameter for the Gaussian Process.",
    )

    parser.add_argument(
        "--gp_length_scale",
        type=float,
        default=1000,
        help="The length scale for the Gaussian Process.",
    )

    parser.add_argument(
        "--gp_sigma_0",
        type=float,
        default=0.001,
        help="The sigma_0 parameter for the Gaussian Process.",
    )

    # N samples to use for GP
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1000,
        help="The number of samples to use for the GP.",
    )

    # Create the output arguments
    parser.add_argument("--output_dir", type=str, help="Path to the output directory")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")

    # Parse the arguments
    args = parser.parse_args()
    print(args)

    main(args)
