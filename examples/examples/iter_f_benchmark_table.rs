//! Benchmark runner for iterated f AIR with multiple configurations.
//!
//! Outputs results as a LaTeX table.
//!
//! Run with: cargo run --release --package p3-examples --example iter_f_benchmark_table

use std::time::Instant;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_examples::iter_f_air::{generate_public_values, generate_trace_rows, IterFAir};
use p3_field::extension::BinomialExtensionField;
use p3_field::Field;
use p3_fri::FriParameters;
use p3_matrix::Matrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{prove, verify, StarkConfig};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

type Val = BabyBear;
type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
type Challenge = BinomialExtensionField<Val, 4>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type Dft = Radix2DitParallel<Val>;
type Pcs = p3_fri::TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
type Config = StarkConfig<Pcs, Challenge, Challenger>;

fn make_config() -> Config {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_params = FriParameters {
        log_blowup: 4,
        log_final_poly_len: 2,
        num_queries: 50,
        commit_proof_of_work_bits: 1,
        query_proof_of_work_bits: 1,
        mmcs: challenge_mmcs,
    };
    let pcs = Pcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);
    Config::new(pcs, challenger)
}

/// Result of a single benchmark run
#[derive(Debug, Clone)]
struct BenchResult {
    num_instances: usize,
    log_iterations: usize,
    trace_height: usize,
    trace_width: usize,
    proof_size_bytes: usize,
    prove_time_ms: f64,
    verify_time_ms: f64,
}

/// Run a benchmark for the given configuration
fn run_benchmark<const N: usize>(log_iterations: usize, num_samples: usize) -> BenchResult {
    let config = make_config();
    let steps = 1usize << log_iterations;

    let mut rng = SmallRng::seed_from_u64(42);
    let initial_values: [u32; N] = core::array::from_fn(|_| rng.random());

    let trace = generate_trace_rows::<Val, N>(initial_values, steps);
    let pis = generate_public_values::<Val, N>(initial_values);
    let air = IterFAir::<N>;

    let trace_height = trace.height();
    let trace_width = trace.width();

    // Warm-up run
    let proof = prove(&config, &air, trace.clone(), &pis);
    let proof_bytes = postcard::to_allocvec(&proof).expect("serialize proof");
    let proof_size_bytes = proof_bytes.len();

    // Verify correctness
    verify(&config, &air, &proof, &pis).expect("verification failed");

    // Benchmark prove time
    let mut prove_times = Vec::with_capacity(num_samples);
    for _ in 0..num_samples {
        let start = Instant::now();
        let _proof = prove(&config, &air, trace.clone(), &pis);
        prove_times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    let prove_time_ms = prove_times.iter().sum::<f64>() / prove_times.len() as f64;

    // Benchmark verify time
    let mut verify_times = Vec::with_capacity(num_samples);
    for _ in 0..num_samples {
        let start = Instant::now();
        verify(&config, &air, &proof, &pis).expect("verification failed");
        verify_times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    let verify_time_ms = verify_times.iter().sum::<f64>() / verify_times.len() as f64;

    BenchResult {
        num_instances: N,
        log_iterations,
        trace_height,
        trace_width,
        proof_size_bytes,
        prove_time_ms,
        verify_time_ms,
    }
}

fn format_size(bytes: usize) -> String {
    if bytes >= 1_000_000 {
        format!("{:.2} MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.1} KB", bytes as f64 / 1_000.0)
    } else {
        format!("{} B", bytes)
    }
}

fn print_latex_table(results: &[BenchResult]) {
    println!("% LaTeX table for iterated f AIR benchmarks");
    println!("\\begin{{table}}[h]");
    println!("\\centering");
    println!("\\caption{{Benchmark results for iterated $f(x) = (x^2 \\bmod 2^{{32}}) \\oplus \\text{{ROTR}}^3(x)$}}");
    println!("\\label{{tab:iter_f_benchmarks}}");
    println!("\\begin{{tabular}}{{|c|c|c|c|c|c|c|}}");
    println!("\\hline");
    println!("\\textbf{{Instances}} & \\textbf{{Iterations}} & \\textbf{{Height}} & \\textbf{{Width}} & \\textbf{{Proof Size}} & \\textbf{{Prove (ms)}} & \\textbf{{Verify (ms)}} \\\\");
    println!("\\hline");

    for r in results {
        println!(
            "{} & $2^{{{}}}$ & {} & {} & {} & {:.1} & {:.2} \\\\",
            r.num_instances,
            r.log_iterations,
            r.trace_height,
            r.trace_width,
            format_size(r.proof_size_bytes),
            r.prove_time_ms,
            r.verify_time_ms
        );
    }

    println!("\\hline");
    println!("\\end{{tabular}}");
    println!("\\end{{table}}");
}

fn main() {
    const NUM_SAMPLES: usize = 5;

    eprintln!("Running iterated f benchmarks...");
    eprintln!("Each configuration will be run {} times and averaged.\n", NUM_SAMPLES);

    let mut results = Vec::new();

    // 1 parallel f, iterations from 2^13 to 2^17
    eprintln!("=== 1 parallel instance ===");
    for log_iter in 13..=17 {
        eprint!("  Running 1x 2^{}... ", log_iter);
        let r = run_benchmark::<1>(log_iter, NUM_SAMPLES);
        eprintln!("prove: {:.1}ms, verify: {:.2}ms", r.prove_time_ms, r.verify_time_ms);
        results.push(r);
    }

    // 4 parallel f, iterations from 2^11 to 2^15
    eprintln!("\n=== 4 parallel instances ===");
    for log_iter in 11..=15 {
        eprint!("  Running 4x 2^{}... ", log_iter);
        let r = run_benchmark::<4>(log_iter, NUM_SAMPLES);
        eprintln!("prove: {:.1}ms, verify: {:.2}ms", r.prove_time_ms, r.verify_time_ms);
        results.push(r);
    }

    // 16 parallel f, iterations from 2^9 to 2^13
    eprintln!("\n=== 16 parallel instances ===");
    for log_iter in 9..=13 {
        eprint!("  Running 16x 2^{}... ", log_iter);
        let r = run_benchmark::<16>(log_iter, NUM_SAMPLES);
        eprintln!("prove: {:.1}ms, verify: {:.2}ms", r.prove_time_ms, r.verify_time_ms);
        results.push(r);
    }

    eprintln!("\n=== LaTeX Table ===\n");
    print_latex_table(&results);
}
