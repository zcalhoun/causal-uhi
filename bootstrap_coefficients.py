import os
import numpy as np
import argparse
import csv
import logging

# SKLEARN modules
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor, kernels

from scipy.optimize import minimize


from src.utils import (
    load_dataset,
    create_folds,
    standardize_data,
    generate_data,
    create_weight_matrix,
)


def main(args):
    # Initialize the same random seed
    np.random.seed(42)

    # Create directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # Initialize the logging
    logging.basicConfig(
        filename=os.path.join(args.output_dir, "output.log"),
        filemode="w",
        level=args.log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info(args)

    ####
    #  Initialize the CSV to store the results
    ####
    headers = ["sample", "train_score", "val_score"]
    # Add the headers for the model coefficients
    std_headers = []
    mean_headers = []
    if args.use_coords:
        for i in range(21):
            headers.append(f"beta_{i}")
            std_headers.append(f"beta_{i}_std")
            mean_headers.append(f"beta_{i}_mean")
    else:
        for i in range(19):
            headers.append(f"beta_{i}")
            std_headers.append(f"beta_{i}_std")
            mean_headers.append(f"beta_{i}_mean")

    headers.extend(["y_mean", "y_std", "model_intercept"])
    # Create header in the CSV
    with open(os.path.join(args.output_dir, "results.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

    with open(os.path.join(args.output_dir, "scale.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(std_headers)

    with open(os.path.join(args.output_dir, "shift.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(mean_headers)

    ####
    # Get the dataset
    ####
    data = load_dataset(args.data_dir, args.window_size)

    # Create the folds
    folds = create_folds(data, args.num_blocks)

    for i in range(args.num_runs):
        # Sample the data
        sample_blocks = np.random.choice(
            range(args.num_blocks), size=args.num_blocks, replace=True
        )
        held_out_blocks = np.setdiff1d(range(args.num_blocks), sample_blocks)

        train_idx = np.where([f in sample_blocks for f in folds])[0]
        val_idx = np.where([f in held_out_blocks for f in folds])[0]

        X_train, y_train = generate_data(
            data,
            train_idx,
            args.ndvi_ls,
            args.albedo_ls,
            args.window_size,
            args.use_coords,
        )
        X_val, y_val = generate_data(
            data,
            val_idx,
            args.ndvi_ls,
            args.albedo_ls,
            args.window_size,
            args.use_coords,
        )

        # Normalize the data
        (
            X_train,
            X_val,
            y_train,
            y_val,
            y_mean,
            y_std,
            x_shift,
            x_scale,
        ) = standardize_data(X_train, X_val, y_train, y_val, scaler=args.scaler)

        if args.iterate:
            # Create and fit the model
            ridge_y = y_train
            # iterating 5 times to stabilize the model
            for _ in range(5):
                lm = Ridge(alpha=args.l2_alpha, fit_intercept=False).fit(
                    X_train, ridge_y
                )

                train_preds = y_train - lm.predict(X_train)

                train_residuals = y_train - train_preds

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

                ridge_y = y_train - U_train
        else:
            # Create and fit the model
            lm = Ridge(alpha=args.l2_alpha).fit(X_train, y_train)

        # Calculate the train and validation score
        train_score = lm.score(X_train, y_train)
        val_score = lm.score(X_val, y_val)

        # Record the results
        row = [i, train_score, val_score]
        row.extend(lm.coef_)
        row.extend([y_mean, y_std])
        row.append(lm.intercept_)

        with open(os.path.join(args.output_dir, "results.csv"), "a") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        with open(os.path.join(args.output_dir, "scale.csv"), "a") as f:
            writer = csv.writer(f)
            writer.writerow(x_scale)

        with open(os.path.join(args.output_dir, "shift.csv"), "a") as f:
            writer = csv.writer(f)
            writer.writerow(x_shift)


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
        description="Bootstrap experiment to retrieve posterior estimates of model coefficients."
    )

    # Create data arguments
    parser.add_argument("--data_dir", type=str, help="Path to the data directory")
    parser.add_argument(
        "--window_size", type=int, default=25, help="Size of the window "
    )
    parser.add_argument("--use_coords", action="store_true", help="Use the coordinates")
    # Create the model arguments
    parser.add_argument(
        "--l2_alpha", type=float, default=10, help="L2 regularization parameter"
    )

    # Create the training arguments
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=50,
        help="Number of folds to use for cross validation",
    )

    parser.add_argument(
        "--num_runs", type=int, default=100, help="Number of bootstrap samples to draw."
    )

    parser.add_argument(
        "--scaler", choices=["standard", "minmax"], default="standard", help="Scaler"
    )

    # Add the start and end ints for the NDVI length scale range
    parser.add_argument(
        "--ndvi_ls",
        type=int,
        help="The LS value to use for NDVI",
        default=7,
    )

    # GP params
    parser.add_argument(
        "--n_samples", type=int, default=1000, help="Number of samples to draw"
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

    # Add the start and end ints for the albedo length scale range
    parser.add_argument(
        "--albedo_ls",
        type=int,
        help="The LS value to use for Albedo",
        default=1,
    )

    # Create the output arguments
    parser.add_argument("--output_dir", type=str, help="Path to the output directory")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")

    # Create flag that determines whether to iterate between GP and Ridge
    parser.add_argument(
        "--iterate",
        action="store_true",
        help="Whether to iterate between GP and Ridge",
    )

    # Parse the arguments
    args = parser.parse_args()
    print(args)

    main(args)
