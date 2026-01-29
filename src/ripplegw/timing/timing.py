"""
Command-line interface for timing gravitational waveform generation in ripple.

This script provides a flexible CLI for benchmarking different waveform approximants
with various configurations including hardware selection, precision, and batch size.
"""

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def get_git_hash():
    """Get the current git commit hash for reproducibility."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def setup_jax_config(use_float64, device):
    """Configure JAX settings for precision and device."""
    jax.config.update("jax_enable_x64", use_float64)

    if device == "cpu":
        jax.config.update("jax_platform_name", "cpu")

    print(f"\n{'=' * 60}")
    print("JAX Configuration")
    print(f"{'=' * 60}")
    print(f"Precision: {'float64' if use_float64 else 'float32'}")
    print(f"Requested device: {device}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
    for d in jax.devices():
        print(f"  Device: {d.device_kind}, Platform: {d.platform}")
    print(f"{'=' * 60}\n")


def generate_bbh_parameters(n_waveforms, seed=42):
    """Generate binary black hole parameters using numpy random sampling.

    Returns parameters for BBH systems including both aligned and precessing spins.
    """
    rng = np.random.default_rng(seed)

    # Mass parameters (5-100 solar masses)
    mass_1 = rng.uniform(10, 100, n_waveforms)
    mass_2 = rng.uniform(0.5, 1.0, n_waveforms) * mass_1  # mass_2 < mass_1

    # Aligned spin magnitudes (-0.99 to 0.99)
    a_1 = rng.uniform(-0.99, 0.99, n_waveforms)
    a_2 = rng.uniform(-0.99, 0.99, n_waveforms)

    # Precessing spin components (-0.99 to 0.99)
    spin_1x = rng.uniform(-0.99, 0.99, n_waveforms)
    spin_1y = rng.uniform(-0.99, 0.99, n_waveforms)
    spin_1z = rng.uniform(-0.99, 0.99, n_waveforms)
    spin_2x = rng.uniform(-0.99, 0.99, n_waveforms)
    spin_2y = rng.uniform(-0.99, 0.99, n_waveforms)
    spin_2z = rng.uniform(-0.99, 0.99, n_waveforms)

    # Distance (100-2000 Mpc)
    luminosity_distance = rng.uniform(100, 2000, n_waveforms)

    # Angles
    theta_jn = rng.uniform(0, np.pi, n_waveforms)  # inclination
    phase = rng.uniform(0, 2 * np.pi, n_waveforms)

    # Time of coalescence
    geocent_time = rng.uniform(0, 1, n_waveforms)

    # Convert to JAX arrays
    params = {
        "mass_1": jnp.array(mass_1),
        "mass_2": jnp.array(mass_2),
        "a_1": jnp.array(a_1),
        "a_2": jnp.array(a_2),
        "spin_1x": jnp.array(spin_1x),
        "spin_1y": jnp.array(spin_1y),
        "spin_1z": jnp.array(spin_1z),
        "spin_2x": jnp.array(spin_2x),
        "spin_2y": jnp.array(spin_2y),
        "spin_2z": jnp.array(spin_2z),
        "luminosity_distance": jnp.array(luminosity_distance),
        "theta_jn": jnp.array(theta_jn),
        "phase": jnp.array(phase),
        "geocent_time": jnp.array(geocent_time),
    }

    return params


def generate_bns_parameters(n_waveforms, seed=42):
    """Generate binary neutron star parameters using numpy random sampling.

    Includes tidal deformability parameters.
    """
    rng = np.random.default_rng(seed)

    # Mass parameters (1-3 solar masses for neutron stars)
    mass_1 = rng.uniform(1.2, 3.0, n_waveforms)
    mass_2 = rng.uniform(0.5, 1.0, n_waveforms) * mass_1  # mass_2 < mass_1

    # Aligned spin magnitudes (neutron stars typically have low spins)
    a_1 = rng.uniform(-0.4, 0.4, n_waveforms)
    a_2 = rng.uniform(-0.4, 0.4, n_waveforms)

    # Tidal deformability parameters (0-5000)
    lambda_1 = rng.uniform(0, 5000, n_waveforms)
    lambda_2 = rng.uniform(0, 5000, n_waveforms)

    # Distance (100-2000 Mpc)
    luminosity_distance = rng.uniform(100, 2000, n_waveforms)

    # Angles
    theta_jn = rng.uniform(0, np.pi, n_waveforms)  # inclination
    phase = rng.uniform(0, 2 * np.pi, n_waveforms)

    # Time of coalescence
    geocent_time = rng.uniform(0, 1, n_waveforms)

    # Convert to JAX arrays
    params = {
        "mass_1": jnp.array(mass_1),
        "mass_2": jnp.array(mass_2),
        "a_1": jnp.array(a_1),
        "a_2": jnp.array(a_2),
        "lambda_1": jnp.array(lambda_1),
        "lambda_2": jnp.array(lambda_2),
        "luminosity_distance": jnp.array(luminosity_distance),
        "theta_jn": jnp.array(theta_jn),
        "phase": jnp.array(phase),
        "geocent_time": jnp.array(geocent_time),
    }

    return params


def time_imrphenomxphm(params, config):
    """Time IMRPhenomXPHM waveform generation."""
    from ripplegw.waveforms import IMRPhenomXPHM

    # Stack parameters for lax.map
    params_stacked = jnp.stack([
        params["mass_1"],
        params["mass_2"],
        params["spin_1x"],
        params["spin_1y"],
        params["spin_1z"],
        params["spin_2x"],
        params["spin_2y"],
        params["spin_2z"],
        params["luminosity_distance"],
        params["theta_jn"],
        params["phase"],
    ], axis=1)

    # Create batched version using lax.map
    def generate_single(p):
        return IMRPhenomXPHM.generate_xphm(
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10],
            config["duration"],
            config["minimum_frequency"],
            config["maximum_frequency"],
            config["reference_frequency"],
        )

    generate_xphm_batched = lambda xs: jax.lax.map(
        generate_single, xs, batch_size=config["batch_size"]
    )

    # Warm-up run (JIT compilation)
    print(f"\n{'=' * 60}")
    print("Warm-up run (JIT compilation)")
    print(f"{'=' * 60}")
    start = time.time()
    hp, hc = generate_xphm_batched(params_stacked)
    hp.block_until_ready()
    hc.block_until_ready()
    warmup_time = time.time() - start
    print(f"Warm-up time (includes JIT): {warmup_time:.3f} s")

    # Timed run
    print(f"\n{'=' * 60}")
    print("Timed run")
    print(f"{'=' * 60}")
    start = time.time()
    hp, hc = generate_xphm_batched(params_stacked)
    hp.block_until_ready()
    hc.block_until_ready()
    exec_time = time.time() - start

    return warmup_time, exec_time


def time_aligned_waveform(waveform_func, params, config):
    """Time aligned-spin waveform generation (IMRPhenomXAS, IMRPhenomD)."""
    from ripplegw import ms_to_Mc_eta

    # Prepare frequency array
    f = jnp.arange(
        config["minimum_frequency"],
        config["maximum_frequency"],
        1.0 / config["duration"],
    )

    # Prepare parameter arrays [Mc, eta, chi1, chi2, D, tc, phic, inclination]
    n_samples = len(params["mass_1"])
    param_arrays = []
    for i in range(n_samples):
        Mc, eta = ms_to_Mc_eta(jnp.array([params["mass_1"][i], params["mass_2"][i]]))
        param_array = jnp.array(
            [
                Mc,
                eta,
                params["a_1"][i],  # chi1
                params["a_2"][i],  # chi2
                params["luminosity_distance"][i],
                params["geocent_time"][i],  # tc
                params["phase"][i],  # phic
                params["theta_jn"][i],  # inclination
            ]
        )
        param_arrays.append(param_array)

    param_arrays = jnp.stack(param_arrays)

    # Create batched version using lax.map
    waveform_batched = lambda xs: jax.lax.map(
        lambda p: waveform_func(f, p, config["reference_frequency"]),
        xs,
        batch_size=config["batch_size"]
    )

    # Warm-up run
    print(f"\n{'=' * 60}")
    print("Warm-up run (JIT compilation)")
    print(f"{'=' * 60}")
    start = time.time()
    hp, hc = waveform_batched(param_arrays)
    hp.block_until_ready()
    hc.block_until_ready()
    warmup_time = time.time() - start
    print(f"Warm-up time (includes JIT): {warmup_time:.3f} s")

    # Timed run
    print(f"\n{'=' * 60}")
    print("Timed run")
    print(f"{'=' * 60}")
    start = time.time()
    hp, hc = waveform_batched(param_arrays)
    hp.block_until_ready()
    hc.block_until_ready()
    exec_time = time.time() - start

    return warmup_time, exec_time


def time_precessing_waveform(waveform_func, params, config):
    """Time precessing waveform generation (IMRPhenomPv2)."""
    from ripplegw import ms_to_Mc_eta

    # Prepare frequency array
    f = jnp.arange(
        config["minimum_frequency"],
        config["maximum_frequency"],
        1.0 / config["duration"],
    )

    # Prepare parameter arrays [Mc, eta, s1x, s1y, s1z, s2x, s2y, s2z, D, tc, phiRef, incl]
    n_samples = len(params["mass_1"])
    param_arrays = []
    for i in range(n_samples):
        Mc, eta = ms_to_Mc_eta(jnp.array([params["mass_1"][i], params["mass_2"][i]]))
        param_array = jnp.array(
            [
                Mc,
                eta,
                params["spin_1x"][i],
                params["spin_1y"][i],
                params["spin_1z"][i],
                params["spin_2x"][i],
                params["spin_2y"][i],
                params["spin_2z"][i],
                params["luminosity_distance"][i],
                params["geocent_time"][i],  # tc
                params["phase"][i],  # phiRef
                params["theta_jn"][i],  # inclination
            ]
        )
        param_arrays.append(param_array)

    param_arrays = jnp.stack(param_arrays)

    # Create batched version using lax.map
    waveform_batched = lambda xs: jax.lax.map(
        lambda p: waveform_func(f, p, config["reference_frequency"]),
        xs,
        batch_size=config["batch_size"]
    )

    # Warm-up run
    print(f"\n{'=' * 60}")
    print("Warm-up run (JIT compilation)")
    print(f"{'=' * 60}")
    start = time.time()
    hp, hc = waveform_batched(param_arrays)
    hp.block_until_ready()
    hc.block_until_ready()
    warmup_time = time.time() - start
    print(f"Warm-up time (includes JIT): {warmup_time:.3f} s")

    # Timed run
    print(f"\n{'=' * 60}")
    print("Timed run")
    print(f"{'=' * 60}")
    start = time.time()
    hp, hc = waveform_batched(param_arrays)
    hp.block_until_ready()
    hc.block_until_ready()
    exec_time = time.time() - start

    return warmup_time, exec_time


def time_bns_waveform(waveform_func, params, config):
    """Time BNS waveform generation (TaylorF2, IMRPhenomD_NRTidalv2)."""
    from ripplegw import ms_to_Mc_eta, lambdas_to_lambda_tildes

    # Prepare frequency array
    f = jnp.arange(
        config["minimum_frequency"],
        config["maximum_frequency"],
        1.0 / config["duration"],
    )

    # Prepare parameter arrays [Mc, eta, chi1, chi2, lambda_tilde, delta_lambda_tilde, D, tc, phic, inclination]
    n_samples = len(params["mass_1"])
    param_arrays = []
    for i in range(n_samples):
        Mc, eta = ms_to_Mc_eta(jnp.array([params["mass_1"][i], params["mass_2"][i]]))
        lambda_tilde, delta_lambda_tilde = lambdas_to_lambda_tildes(
            jnp.array(
                [
                    params["lambda_1"][i],
                    params["lambda_2"][i],
                    params["mass_1"][i],
                    params["mass_2"][i],
                ]
            )
        )
        param_array = jnp.array(
            [
                Mc,
                eta,
                params["a_1"][i],  # chi1
                params["a_2"][i],  # chi2
                lambda_tilde,
                delta_lambda_tilde,
                params["luminosity_distance"][i],
                params["geocent_time"][i],  # tc
                params["phase"][i],  # phic
                params["theta_jn"][i],  # inclination
            ]
        )
        param_arrays.append(param_array)

    param_arrays = jnp.stack(param_arrays)

    # Create batched version using lax.map
    waveform_batched = lambda xs: jax.lax.map(
        lambda p: waveform_func(f, p, config["reference_frequency"]),
        xs,
        batch_size=config["batch_size"]
    )

    # Warm-up run
    print(f"\n{'=' * 60}")
    print("Warm-up run (JIT compilation)")
    print(f"{'=' * 60}")
    start = time.time()
    hp, hc = waveform_batched(param_arrays)
    hp.block_until_ready()
    hc.block_until_ready()
    warmup_time = time.time() - start
    print(f"Warm-up time (includes JIT): {warmup_time:.3f} s")

    # Timed run
    print(f"\n{'=' * 60}")
    print("Timed run")
    print(f"{'=' * 60}")
    start = time.time()
    hp, hc = waveform_batched(param_arrays)
    hp.block_until_ready()
    hc.block_until_ready()
    exec_time = time.time() - start

    return warmup_time, exec_time


def run_timing(args):
    """Main timing function that orchestrates the benchmark."""
    # Setup configuration
    config = {
        "waveform": args.waveform,
        "device": args.device,
        "n_waveforms": args.n_waveforms,
        "batch_size": args.batch_size,
        "precision": "float64" if args.float64 else "float32",
        "duration": args.duration,
        "minimum_frequency": args.f_min,
        "maximum_frequency": args.f_max,
        "reference_frequency": args.f_ref,
        "timestamp": datetime.now().isoformat(),
        "git_hash": get_git_hash(),
    }

    # Setup JAX
    setup_jax_config(args.float64, args.device)

    # Print configuration
    print(f"{'=' * 60}")
    print("Timing Configuration")
    print(f"{'=' * 60}")
    print(f"Waveform: {args.waveform}")
    print(f"Number of waveforms: {args.n_waveforms}")
    print(f"Duration: {args.duration} s")
    print(f"Frequency range: {args.f_min} - {args.f_max} Hz")
    print(f"Reference frequency: {args.f_ref} Hz")
    print(f"Git hash: {config['git_hash']}")
    print(f"{'=' * 60}\n")

    # Generate parameters based on waveform type
    waveform_type = get_waveform_type(args.waveform)

    if waveform_type == "bns":
        params = generate_bns_parameters(args.n_waveforms)
    else:
        params = generate_bbh_parameters(args.n_waveforms)

    print(f"Generated {args.n_waveforms} parameter sets")
    print(f"Parameter keys: {list(params.keys())}\n")

    # Run timing based on waveform
    if args.waveform == "IMRPhenomXPHM":
        warmup_time, exec_time = time_imrphenomxphm(params, config)
    elif args.waveform in ["IMRPhenomXAS", "IMRPhenomD"]:
        from ripplegw.waveforms.IMRPhenomXAS import gen_IMRPhenomXAS_hphc
        from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD_hphc

        waveform_func = {
            "IMRPhenomXAS": gen_IMRPhenomXAS_hphc,
            "IMRPhenomD": gen_IMRPhenomD_hphc,
        }[args.waveform]

        warmup_time, exec_time = time_aligned_waveform(waveform_func, params, config)
    elif args.waveform == "IMRPhenomPv2":
        from ripplegw.waveforms.IMRPhenomPv2 import gen_IMRPhenomPv2_hphc

        warmup_time, exec_time = time_precessing_waveform(
            gen_IMRPhenomPv2_hphc, params, config
        )
    elif args.waveform in ["TaylorF2", "IMRPhenomD_NRTidalv2"]:
        from ripplegw.waveforms.TaylorF2 import gen_TaylorF2_hphc

        # Note: IMRPhenomD_NRTidalv2 not yet implemented in this timing script
        warmup_time, exec_time = time_bns_waveform(gen_TaylorF2_hphc, params, config)
    else:
        raise ValueError(f"Unknown waveform: {args.waveform}")

    # Print results
    print(f"\n{'=' * 60}")
    print("Timing Results")
    print(f"{'=' * 60}")
    print(f"Warm-up time (includes JIT): {warmup_time:.6f} s")
    print(f"Execution time for {args.n_waveforms} waveforms: {exec_time:.6f} s")
    print(f"Time per waveform: {exec_time / args.n_waveforms * 1000:.3f} ms")
    print(f"Waveforms per second: {args.n_waveforms / exec_time:.1f}")
    print(f"{'=' * 60}\n")

    # Save results
    results = {
        **config,
        "warmup_time_s": float(warmup_time),
        "execution_time_s": float(exec_time),
        "time_per_waveform_ms": float(exec_time / args.n_waveforms * 1000),
        "waveforms_per_second": float(args.n_waveforms / exec_time),
    }

    # Save to JSON
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(
            f"timing_{args.waveform}.json"
        )

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")


def get_waveform_type(waveform):
    """Determine if waveform is BBH or BNS."""
    bns_waveforms = ["TaylorF2", "IMRPhenomD_NRTidalv2"]
    return "bns" if waveform in bns_waveforms else "bbh"


def main():
    """Parse arguments and run timing benchmark."""
    parser = argparse.ArgumentParser(
        description="Time gravitational waveform generation in ripple",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "waveform",
        type=str,
        choices=[
            "IMRPhenomXPHM",
            "IMRPhenomXAS",
            "IMRPhenomD",
            "IMRPhenomPv2",
            "TaylorF2",
            "IMRPhenomD_NRTidalv2",
        ],
        help="Waveform approximant to time",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        default="gpu",
        help="Hardware device to use",
    )

    parser.add_argument(
        "--n-waveforms",
        type=int,
        default=int(2e4),
        help="Number of waveforms to generate (for vmapping)",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(1000),
        help="Number of waveforms to generate (for jax lax map)",
    )

    parser.add_argument(
        "--float64",
        action="store_true",
        help="Use float64 precision (default is float32)",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=4.0,
        help="Duration of the waveform in seconds",
    )

    parser.add_argument(
        "--f-min", type=float, default=20.0, help="Minimum frequency in Hz"
    )

    parser.add_argument(
        "--f-max", type=float, default=2048.0, help="Maximum frequency in Hz"
    )

    parser.add_argument(
        "--f-ref", type=float, default=50.0, help="Reference frequency in Hz"
    )

    parser.add_argument(
        "--output", type=str, help="Output JSON file path (default: auto-generated)"
    )

    args = parser.parse_args()

    run_timing(args)


if __name__ == "__main__":
    main()
