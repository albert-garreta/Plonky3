use core::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use p3_field::PrimeCharacteristicRing;
use p3_mersenne_31::Mersenne31;
use p3_sha256::{Sha256, Sha256Compress};
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction, SerializingHasher};

pub fn criterion_benchmark(c: &mut Criterion) {
    sha256_compression(c);
    sha256_u8_hash(c);
    sha256_field_32_hash(c);
}

pub fn sha256_compression(c: &mut Criterion) {
    let sha256_compress = Sha256Compress;
    let input: [[u8; 32]; 2] = [[0u8; 32]; 2];

    let mut group = c.benchmark_group("sha256 compression");
    group.throughput(Throughput::Bytes(64));
    group.bench_function("sha256 compress [[u8; 32]; 2]", |b| {
        b.iter(|| sha256_compress.compress(black_box(input)));
    });

    const NUM_COMPRESSIONS: usize = 32;
    group.throughput(Throughput::Bytes((64 * NUM_COMPRESSIONS) as u64));
    group.bench_function("sha256 compress x32", |b| {
        b.iter(|| {
            for _ in 0..NUM_COMPRESSIONS {
                black_box(sha256_compress.compress(black_box(input)));
            }
        });
    });
    group.finish();
}

pub fn sha256_u8_hash(c: &mut Criterion) {
    const BYTES_PER_HASH: usize = 6400;
    let input = vec![0u8; BYTES_PER_HASH];

    let sha256 = Sha256;

    let mut group = c.benchmark_group("sha256 u8 hash");
    group.throughput(Throughput::Bytes(BYTES_PER_HASH as u64));
    group.bench_function("sha256 u8 hash_iter", |b| {
        b.iter(|| sha256.hash_iter(black_box(input.clone())));
    });
    group.bench_function("sha256 u8 hash_iter_slices", |b| {
        b.iter(|| sha256.hash_iter_slices(black_box([input.as_slice()])));
    });
    group.finish();
}

pub fn sha256_field_32_hash(c: &mut Criterion) {
    type F = Mersenne31;
    const ELEMS_PER_HASH: usize = 100;
    const BYTES_PER_HASH: usize = size_of::<F>() * ELEMS_PER_HASH;
    let input = vec![F::ZERO; ELEMS_PER_HASH];

    type FieldHash = SerializingHasher<Sha256>;
    let field_hash = FieldHash::new(Sha256);

    let mut group = c.benchmark_group("sha256 field 32 hash");
    group.throughput(Throughput::Bytes(BYTES_PER_HASH as u64));
    group.bench_function("sha256 field 32 hash_slice", |b| {
        b.iter(|| {
            <FieldHash as CryptographicHasher<F, [u8; 32]>>::hash_slice(
                &field_hash,
                black_box(&input),
            )
        });
    });
    group.bench_function("sha256 field 32 hash_iter", |b| {
        b.iter(|| {
            <FieldHash as CryptographicHasher<F, [u8; 32]>>::hash_iter(
                &field_hash,
                black_box(input.iter().copied()),
            )
        });
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
