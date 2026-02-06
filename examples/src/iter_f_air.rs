use core::array;

use p3_air::utils::{pack_bits_le, u64_to_bits_le};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;

const NUM_BITS: usize = 32;
const LIMB_BITS: usize = 16;
const T1_HI_BITS: usize = 17; // 2*x0*x1 can be up to 2*(2^16-1)^2 ≈ 2^33, so t1_hi needs 17 bits
/// Width of a single instance's columns
const INSTANCE_WIDTH: usize = NUM_BITS * 2 + LIMB_BITS * 2 + T1_HI_BITS + 1;

/// Compute the total trace width for N parallel instances
pub const fn trace_width<const N: usize>() -> usize {
    INSTANCE_WIDTH * N
}

/// AIR for iterating `f(x) = (x^2 mod 2^32) XOR ROTR^3(x)` in parallel.
///
/// The `NUM_INSTANCES` const generic allows computing multiple independent
/// instances of the iterated function in parallel within the same trace.
///
/// Public values (2 * NUM_INSTANCES):
/// - For each instance i: `x0_low[i]`, `x0_high[i]` (low/high 16 bits of initial input)
#[derive(Debug, Clone, Copy)]
pub struct IterFAir<const NUM_INSTANCES: usize>;

/// Type alias for convenience - single instance
pub type IterFAirSingle = IterFAir<1>;

impl<F, const N: usize> BaseAir<F> for IterFAir<N> {
    fn width(&self) -> usize {
        trace_width::<N>()
    }
}

impl<AB: AirBuilderWithPublicValues, const N: usize> Air<AB> for IterFAir<N> {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("Matrix is empty?");
        let next = main.row_slice(1).expect("Matrix only has 1 row?");

        let two_pow_16 = AB::Expr::from_u64(1u64 << LIMB_BITS);

        // Apply constraints for each parallel instance
        for inst in 0..N {
            let offset = inst * INSTANCE_WIDTH;

            // Extract columns for this instance
            let x_bits = &local[offset..offset + NUM_BITS];
            let sq_bits = &local[offset + NUM_BITS..offset + 2 * NUM_BITS];
            let x0_sq_hi_bits = &local[offset + 2 * NUM_BITS..offset + 2 * NUM_BITS + LIMB_BITS];
            let t1_low_bits =
                &local[offset + 2 * NUM_BITS + LIMB_BITS..offset + 2 * NUM_BITS + 2 * LIMB_BITS];
            let t1_hi_bits = &local
                [offset + 2 * NUM_BITS + 2 * LIMB_BITS..offset + 2 * NUM_BITS + 2 * LIMB_BITS + T1_HI_BITS];
            let s1_carry = local[offset + INSTANCE_WIDTH - 1].clone();

            let next_x_bits = &next[offset..offset + NUM_BITS];

            // Range check all bit columns.
            for bit in x_bits.iter() {
                builder.assert_bool(bit.clone());
            }
            for bit in sq_bits.iter() {
                builder.assert_bool(bit.clone());
            }
            for bit in x0_sq_hi_bits.iter() {
                builder.assert_bool(bit.clone());
            }
            for bit in t1_low_bits.iter() {
                builder.assert_bool(bit.clone());
            }
            for bit in t1_hi_bits.iter() {
                builder.assert_bool(bit.clone());
            }
            builder.assert_bool(s1_carry.clone());

            // Pack x, square, and helper limbs.
            let x0: AB::Expr = pack_bits_le(x_bits[..LIMB_BITS].iter().cloned());
            let x1: AB::Expr = pack_bits_le(x_bits[LIMB_BITS..].iter().cloned());
            let s0: AB::Expr = pack_bits_le(sq_bits[..LIMB_BITS].iter().cloned());
            let s1: AB::Expr = pack_bits_le(sq_bits[LIMB_BITS..].iter().cloned());
            let x0_sq_hi: AB::Expr = pack_bits_le(x0_sq_hi_bits.iter().cloned());
            let t1_low: AB::Expr = pack_bits_le(t1_low_bits.iter().cloned());

            // Enforce the 32-bit square decomposition using 16-bit limbs.
            // t0 = x0^2 = s0 + 2^16 * x0_sq_hi
            let t0 = x0.clone() * x0.clone();
            builder.assert_eq(t0, s0.clone() + two_pow_16.clone() * x0_sq_hi.clone());

            // t1 = 2 * x0 * x1 = t1_low + 2^16 * t1_hi
            let t1 = x0.clone() * x1.clone() * AB::Expr::TWO;
            let t1_hi: AB::Expr = pack_bits_le(t1_hi_bits.iter().cloned());
            builder.assert_eq(t1, t1_low.clone() + two_pow_16.clone() * t1_hi);

            // s1 = (x0_sq_hi + t1_low) mod 2^16
            builder.assert_eq(
                x0_sq_hi + t1_low,
                s1 + two_pow_16.clone() * s1_carry.into(),
            );

            // Transition: next.x_bits = sq_bits XOR ROTR^3(x_bits)
            let mut when_transition = builder.when_transition();
            for i in 0..NUM_BITS {
                let rot_i = x_bits[(i + 3) % NUM_BITS].clone();
                let xor = sq_bits[i].clone().into().xor(&rot_i.into());
                when_transition.assert_eq(next_x_bits[i].clone(), xor);
            }

            // First row input equals public input for this instance.
            // Public values layout: [x0_low_0, x0_high_0, x0_low_1, x0_high_1, ...]
            let pis = builder.public_values();
            let pi_offset = inst * 2;
            let x0_low_pi = pis[pi_offset].clone();
            let x0_high_pi = pis[pi_offset + 1].clone();
            let mut when_first = builder.when_first_row();
            when_first.assert_eq(x0, x0_low_pi);
            when_first.assert_eq(x1, x0_high_pi);
        }
    }
}

/// Single instance row data (used internally for trace generation)
#[derive(Clone, Copy, Debug)]
struct IterFInstanceRow<F> {
    x_bits: [F; NUM_BITS],
    sq_bits: [F; NUM_BITS],
    x0_sq_hi_bits: [F; LIMB_BITS],
    t1_low_bits: [F; LIMB_BITS],
    t1_hi_bits: [F; T1_HI_BITS],
    s1_carry: F,
}

/// Compute the columns for a single instance given input x
fn compute_instance_columns<F: PrimeField64>(x: u32) -> (IterFInstanceRow<F>, u32) {
    let sq = x.wrapping_mul(x);
    let rot = x.rotate_right(3);
    let out = sq ^ rot;

    let x_bits = bits_le_32::<F>(x);
    let sq_bits = bits_le_32::<F>(sq);

    let x0_low = (x & 0xffff) as u64;
    let x1_high = (x >> 16) as u64;
    let t0 = x0_low * x0_low;
    let x0_sq_hi = ((t0 >> 16) & 0xffff) as u16;

    let t1 = 2 * x0_low * x1_high;
    let t1_low = (t1 & 0xffff) as u16;
    let t1_hi = (t1 >> 16) as u32;

    let s1_carry = ((x0_sq_hi as u64 + t1_low as u64) >> 16) & 1;

    let row = IterFInstanceRow {
        x_bits,
        sq_bits,
        x0_sq_hi_bits: bits_le_16::<F>(x0_sq_hi),
        t1_low_bits: bits_le_16::<F>(t1_low),
        t1_hi_bits: bits_le_17::<F>(t1_hi),
        s1_carry: F::from_u64(s1_carry),
    };

    (row, out)
}

/// Write instance columns to a row at the given offset
fn write_instance_to_row<F: Copy>(row: &mut [F], offset: usize, inst: &IterFInstanceRow<F>) {
    let mut idx = offset;
    for &bit in &inst.x_bits {
        row[idx] = bit;
        idx += 1;
    }
    for &bit in &inst.sq_bits {
        row[idx] = bit;
        idx += 1;
    }
    for &bit in &inst.x0_sq_hi_bits {
        row[idx] = bit;
        idx += 1;
    }
    for &bit in &inst.t1_low_bits {
        row[idx] = bit;
        idx += 1;
    }
    for &bit in &inst.t1_hi_bits {
        row[idx] = bit;
        idx += 1;
    }
    row[idx] = inst.s1_carry;
}

/// Generate a trace for N parallel instances, each running for `num_steps`.
///
/// Each row stores N independent instances of the function iteration.
/// The `initial_values` array contains the starting x value for each instance.
pub fn generate_trace_rows<F: PrimeField64, const N: usize>(
    initial_values: [u32; N],
    num_steps: usize,
) -> RowMajorMatrix<F> {
    let width = trace_width::<N>();
    let mut trace = RowMajorMatrix::new(F::zero_vec(num_steps * width), width);

    // Current x value for each instance
    let mut xs = initial_values;

    for step in 0..num_steps {
        let row_start = step * width;
        let row = &mut trace.values[row_start..row_start + width];

        for inst in 0..N {
            let (inst_row, next_x) = compute_instance_columns::<F>(xs[inst]);
            write_instance_to_row(row, inst * INSTANCE_WIDTH, &inst_row);
            xs[inst] = next_x;
        }
    }

    trace
}

/// Generate public values for the parallel instances.
///
/// Returns a vector of 2*N field elements: [x0_low_0, x0_high_0, x0_low_1, x0_high_1, ...]
pub fn generate_public_values<F: PrimeField64, const N: usize>(initial_values: [u32; N]) -> Vec<F> {
    let mut pis = Vec::with_capacity(2 * N);
    for x0 in initial_values {
        pis.push(F::from_u64((x0 & 0xffff) as u64));
        pis.push(F::from_u64((x0 >> 16) as u64));
    }
    pis
}

#[inline]
fn bits_le_32<R: PrimeCharacteristicRing>(val: u32) -> [R; NUM_BITS] {
    let bits = u64_to_bits_le::<R>(val as u64);
    array::from_fn(|i| bits[i].clone())
}

#[inline]
fn bits_le_16<R: PrimeCharacteristicRing>(val: u16) -> [R; LIMB_BITS] {
    let bits = u64_to_bits_le::<R>(val as u64);
    array::from_fn(|i| bits[i].clone())
}

#[inline]
fn bits_le_17<R: PrimeCharacteristicRing>(val: u32) -> [R; T1_HI_BITS] {
    let bits = u64_to_bits_le::<R>(val as u64);
    array::from_fn(|i| bits[i].clone())
}
