import os
import numpy as np
import csv
import argparse
import logging

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from scipy.interpolate import griddata

from scipy.optimize import minimize

from src.utils import load_dataset, generate_data


def main(args):
    # Check if the output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Create the logger
    create_logger(args.output_dir, args.log_level)
    logging.info(args)

    # Load the data
    data = load_dataset(args.data_dir, args.window_size)

    idx = range(len(data.coords))
    X, y = generate_data(
        data, idx, args.ndvi_ls, args.albedo_ls, args.window_size, args.use_coords
    )

    # Standardize the data
    X, y, y_scale, y_shift = standardize_data(X, y)

    # Initialize the CSV to store results
    headers = ["iteration", "ridge_r2", "gp_r2", "total_r2"]

    for i in range(19):
        headers.append(f"beta_{i}")

    headers.append("intercept")
    headers.extend(["matern const", "matern_ls", "dp const", "dot_prod_sigma", "noise"])

    with open(os.path.join(args.output_dir, "results.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

    ridge_output = y
    for i in range(args.max_iter):
        row = [i]
        logging.info(f"Iteration {i}.")
        logging.info("Fitting ridge")
        # Fit the ridge regression model
        ridge = Ridge(alpha=args.l2_alpha, fit_intercept=args.fit_intercept)
        ridge.fit(X, ridge_output)

        # Get the predictions
        y_pred = ridge.predict(X)

        # Compute the metrics
        ridge_r2 = r2_score(y, y_pred)
        row.append(ridge_r2)
        # Get residuals
        residuals = y - y_pred
        logging.info("Fitting GP.")
        # fit the gaussian process
        sample_points, sample_residuals = generate_samples(
            data.coords, residuals, args.n_samples
        )

        # Fit the GP
        gp = fit_gp(
            sample_points,
            sample_residuals,
            args.gp_constant_1,
            args.gp_length_scale,
            args.gp_constant_2,
            args.gp_sigma_0,
            args.gp_noise,
        )

        U = gp.predict(data.coords)

        gp_r2 = r2_score(residuals, U)
        row.append(gp_r2)
        total_r2 = r2_score(y, y_pred + U)
        row.append(total_r2)
        row.extend(ridge.coef_)
        row.append(ridge.intercept_)

        # Get the kernel parameters
        kp = gp.kernel_.get_params()
        row.append(np.sqrt(kp["k1__k1__constant_value"]))
        row.append(kp["k1__k2__length_scale"])
        row.append(np.sqrt(kp["k2__k1__constant_value"]))
        row.append(kp["k2__k2__sigma_0"])
        row.append(gp.alpha)

        # Save the results
        with open(os.path.join(args.output_dir, "results.csv"), "a") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        ridge_output = y - U


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

    # # If noise is negative, then automatically fit to the noise term
    # if noise < 0:
    #     kernel = kernel + kernels.WhiteKernel(noise_level=0.1)
    # else:
    #     kernel = kernel + kernels.WhiteKernel(
    #         noise_level=noise, noise_level_bounds="fixed"
    #     )

    gpr = GaussianProcessRegressor(kernel=kernel, optimizer=optimizer, alpha=noise)

    gpr.fit(points, residuals)

    return gpr


def generate_samples(coords, residuals, N):
    sample_idx = np.random.choice(np.arange(coords.shape[0]), N, replace=False)
    sample_points = coords[sample_idx, :]
    sample_residuals = residuals[sample_idx]

    return sample_points, sample_residuals


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


def standardize_data(X, y):
    x_shift = X.min(axis=0)
    x_scale = X.max(axis=0) - X.min(axis=0)

    X = (X - x_shift) / (x_scale + 1e-8)

    # Standardize the labels
    y_mean = y.mean()
    y_std = y.std()

    y = (y - y_mean) / y_std

    return X, y, y_mean, y_std


def create_logger(output_dir, log_level):
    logging.basicConfig(
        filename=os.path.join(output_dir, "output.log"),
        filemode="w",
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Jointly train the model with unobserved confounding."
    )

    # Add an argument for the data directory
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/durham/",
        help="The directory containing the data.",
    )

    # Add an argument for the output directory
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/joint/",
        help="The directory to save the output.",
    )

    # Add a gp term for the constant
    parser.add_argument(
        "--gp_constant_1",
        type=float,
        default=0.5,
        help="The constant parameter for the Gaussian Process.",
    )
    parser.add_argument(
        "--gp_constant_2",
        type=float,
        default=0.5,
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
        default=150,
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

    # Add a logging level
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="The logging level.",
    )

    # Create arguments for data
    # window size
    parser.add_argument(
        "--window_size",
        type=int,
        default=25,
        help="The window size to use for the data.",
    )
    # ndvi_ls
    parser.add_argument(
        "--ndvi_ls",
        type=int,
        default=4,
        help="The length scale to use for the NDVI kernel.",
    )
    # albedo_ls
    parser.add_argument(
        "--albedo_ls",
        type=int,
        default=3,
        help="The length scale to use for the Albedo kernel.",
    )
    # use_coords
    parser.add_argument(
        "--use_coords",
        action="store_true",
        help="Whether to use the coordinates.",
    )
    # l2_alpha
    parser.add_argument(
        "--l2_alpha",
        type=float,
        default=100.0,
        help="The L2 regularization parameter.",
    )
    # fit_intercept
    parser.add_argument(
        "--fit_intercept",
        action="store_true",
        help="Whether to fit an intercept.",
    )

    # Max iterations
    parser.add_argument(
        "--max_iter",
        type=int,
        default=20,
        help="The maximum number of iterations.",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Run the main function
    main(args)
