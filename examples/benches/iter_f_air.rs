use criterion::{criterion_group, criterion_main, Criterion};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_examples::iter_f_air::{IterFAir, generate_trace_rows};
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_fri::FriParameters;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{StarkConfig, prove, verify};
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

fn make_two_adic_config(log_final_poly_len: usize) -> Config {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_params = FriParameters {
        log_blowup: 2,
        log_final_poly_len,
        num_queries: 2,
        commit_proof_of_work_bits: 1,
        query_proof_of_work_bits: 1,
        mmcs: challenge_mmcs,
    };
    let pcs = Pcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);
    Config::new(pcs, challenger)
}

fn sample_case() -> (IterFAir, RowMajorMatrix<Val>, Vec<Val>) {
    let mut rng = SmallRng::seed_from_u64(1);
    let x0: u32 = rng.random();
    let trace = generate_trace_rows::<Val>(x0);
    let pis = vec![Val::from_u64((x0 & 0xffff) as u64), Val::from_u64((x0 >> 16) as u64)];
    (IterFAir, trace, pis)
}

fn bench_iter_f_trace(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(1);
    c.bench_function("iter_f/trace_gen_2^10", |b| {
        b.iter(|| {
            let x0: u32 = rng.random();
            let trace = generate_trace_rows::<Val>(x0);
            black_box(trace);
        })
    });
}

fn bench_iter_f_prove_verify(c: &mut Criterion) {
    let (air, trace, pis) = sample_case();
    let config = make_two_adic_config(2);

    let proof = prove(&config, &air, trace.clone(), &pis);
    let proof_bytes = postcard::to_allocvec(&proof).expect("serialize proof");
    println!("iter_f/proof_size_bytes: {}", proof_bytes.len());

    c.bench_function("iter_f/prove", |b| {
        b.iter(|| {
            let proof = prove(&config, &air, trace.clone(), &pis);
            black_box(proof);
        })
    });

    c.bench_function("iter_f/verify", |b| {
        b.iter(|| {
            verify(&config, &air, &proof, &pis).expect("verification failed");
        })
    });
}

criterion_group!(benches, bench_iter_f_trace, bench_iter_f_prove_verify);
criterion_main!(benches);
