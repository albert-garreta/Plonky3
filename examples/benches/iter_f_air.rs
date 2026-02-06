use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_bn254::Bn254;
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_examples::iter_f_air::{IterFAir, generate_trace_rows, generate_public_values};
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField};
use p3_fri::FriParameters;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{StarkConfig, prove, verify};
use p3_util::log2_strict_usize;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::hint::black_box;

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

const G_STEP_LOGS: [usize; 5] = [13, 14, 15, 16, 17];

fn make_two_adic_config(log_final_poly_len: usize) -> Config {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_params = FriParameters {
        log_blowup: 4,
        log_final_poly_len,
        num_queries: 50,
        commit_proof_of_work_bits: 1,
        query_proof_of_work_bits: 1,
        mmcs: challenge_mmcs,
    };
    let pcs = Pcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);
    Config::new(pcs, challenger)
}

fn sample_case<const N: usize>(num_steps: usize) -> (IterFAir<N>, RowMajorMatrix<Val>, Vec<Val>) {
    let mut rng = SmallRng::seed_from_u64(1);
    let initial_values: [u32; N] = core::array::from_fn(|_| rng.random());
    let trace = generate_trace_rows::<Val, N>(initial_values, num_steps);
    let pis = generate_public_values::<Val, N>(initial_values);
    (IterFAir::<N>, trace, pis)
}

fn bench_iter_f_trace(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(1);
    let mut group = c.benchmark_group("iter_f_trace");
    for log_steps in G_STEP_LOGS {
        let steps = 1usize << log_steps;
        group.bench_with_input(BenchmarkId::from_parameter(format!("2^{log_steps}")), &steps, |b, &steps| {
            b.iter(|| {
                let x0: u32 = rng.random();
                let trace = generate_trace_rows::<Val, 1>([x0], steps);
                black_box(trace);
            })
        });
    }
    group.finish();
}

fn bench_iter_f_prove_verify(c: &mut Criterion) {
    let mut prove_group = c.benchmark_group("iter_f_prove");
    for log_steps in G_STEP_LOGS {
        let steps = 1usize << log_steps;
        let (air, trace, pis) = sample_case::<1>(steps);
        let config = make_two_adic_config(2);

        eprintln!("iter_f/2^{}/trace_height: {}", log_steps, trace.height());
        eprintln!("iter_f/2^{}/log_n: {}", log_steps, log2_strict_usize(trace.height()));
        eprintln!("iter_f/2^{}/trace_width: {}", log_steps, trace.width());

        let proof = prove(&config, &air, trace.clone(), &pis);
        let proof_bytes = postcard::to_allocvec(&proof).expect("serialize proof");
        eprintln!("iter_f/2^{}/proof_size_bytes: {}", log_steps, proof_bytes.len());

        prove_group.bench_with_input(BenchmarkId::from_parameter(format!("2^{log_steps}")), &steps, |b, _| {
            b.iter(|| {
                let proof = prove(&config, &air, trace.clone(), &pis);
                black_box(proof);
            })
        });
    }
    prove_group.finish();

    let mut verify_group = c.benchmark_group("iter_f_verify");
    for log_steps in G_STEP_LOGS {
        let steps = 1usize << log_steps;
        let (air, trace, pis) = sample_case::<1>(steps);
        let config = make_two_adic_config(2);
        let proof = prove(&config, &air, trace.clone(), &pis);

        verify_group.bench_with_input(BenchmarkId::from_parameter(format!("2^{log_steps}")), &steps, |b, _| {
            b.iter(|| {
                verify(&config, &air, &proof, &pis).expect("verification failed");
            })
        });
    }
    verify_group.finish();
}

#[inline]
fn f_u32(x: u32) -> u32 {
    let sq = x.wrapping_mul(x);
    let rot = x.rotate_right(3);
    sq ^ rot
}

#[inline]
fn low_u32_from_bn254(x: &Bn254) -> u32 {
    let digits = x.as_canonical_biguint().to_u64_digits();
    digits.first().copied().unwrap_or(0) as u32
}

#[inline]
fn g_bn254(x: Bn254) -> Bn254 {
    let fx = f_u32(low_u32_from_bn254(&x));
    let fx_field = Bn254::from_u64(fx as u64);
    fx_field.square()
}

fn bench_iter_g_bn254(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(2);
    let mut group = c.benchmark_group("iter_g_bn254");
    for log_steps in G_STEP_LOGS {
        let steps = 1usize << log_steps;
        group.bench_with_input(BenchmarkId::from_parameter(format!("2^{log_steps}")), &steps, |b, &steps| {
            b.iter(|| {
                let mut x = Bn254::from_u64(rng.random::<u32>() as u64);
                for _ in 0..steps {
                    x = g_bn254(x);
                }
                let _ = black_box(x);
            })
        });
    }
    group.finish();
}

fn bench_iter_f_parallel(c: &mut Criterion) {
    let mut prove_group = c.benchmark_group("iter_f_parallel_prove");
    
    // 32 parallel instances, 2^8 iterations
    const NUM_INSTANCES: usize = 32;
    const LOG_STEPS: usize = 8;
    let steps = 1usize << LOG_STEPS;
    
    let (air, trace, pis) = sample_case::<NUM_INSTANCES>(steps);
    let config = make_two_adic_config(2);
    
    eprintln!("iter_f_parallel/instances: {}", NUM_INSTANCES);
    eprintln!("iter_f_parallel/2^{}/trace_height: {}", LOG_STEPS, trace.height());
    eprintln!("iter_f_parallel/2^{}/log_n: {}", LOG_STEPS, log2_strict_usize(trace.height()));
    eprintln!("iter_f_parallel/2^{}/trace_width: {}", LOG_STEPS, trace.width());
    
    let proof = prove(&config, &air, trace.clone(), &pis);
    let proof_bytes = postcard::to_allocvec(&proof).expect("serialize proof");
    eprintln!("iter_f_parallel/2^{}/proof_size_bytes: {}", LOG_STEPS, proof_bytes.len());
    
    prove_group.bench_with_input(
        BenchmarkId::from_parameter(format!("{}x_2^{}", NUM_INSTANCES, LOG_STEPS)),
        &steps,
        |b, _| {
            b.iter(|| {
                let proof = prove(&config, &air, trace.clone(), &pis);
                black_box(proof);
            })
        },
    );
    prove_group.finish();
    
    let mut verify_group = c.benchmark_group("iter_f_parallel_verify");
    verify_group.bench_with_input(
        BenchmarkId::from_parameter(format!("{}x_2^{}", NUM_INSTANCES, LOG_STEPS)),
        &steps,
        |b, _| {
            b.iter(|| {
                verify(&config, &air, &proof, &pis).expect("verification failed");
            })
        },
    );
    verify_group.finish();
}

criterion_group!(
    benches,
    bench_iter_f_trace,
    bench_iter_f_prove_verify,
    bench_iter_g_bn254,
    bench_iter_f_parallel
);
criterion_main!(benches);
