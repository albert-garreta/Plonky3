use core::array;
use core::borrow::Borrow;

use p3_air::utils::{pack_bits_le, u64_to_bits_le};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;

const NUM_BITS: usize = 32;
const LIMB_BITS: usize = 16;
const NUM_STEPS: usize = 1 << 13;
const TRACE_WIDTH: usize = NUM_BITS * 4 + LIMB_BITS * 2 + 2;

/// AIR for iterating `f(x) = (x^2 mod 2^32) XOR ROTR^3(x)` for `2^13` steps.
///
/// Public values (2):
/// - `x0_low`: low 16 bits of the initial input `x_0`
/// - `x0_high`: high 16 bits of the initial input `x_0`
#[derive(Debug, Clone, Copy)]
pub struct IterFAir;

impl<F> BaseAir<F> for IterFAir {
    fn width(&self) -> usize {
        TRACE_WIDTH
    }
}

impl<AB: AirBuilderWithPublicValues> Air<AB> for IterFAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("Matrix is empty?"),
            main.row_slice(1).expect("Matrix only has 1 row?"),
        );
        let local: &IterFRow<AB::Var> = (*local).borrow();
        let next: &IterFRow<AB::Var> = (*next).borrow();

        // Range check all bit columns.
        builder.assert_bools(local.x_bits.clone());
        builder.assert_bools(local.sq_bits.clone());
        builder.assert_bools(local.rot_bits.clone());
        builder.assert_bools(local.out_bits.clone());
        builder.assert_bools(local.x0_sq_hi_bits.clone());
        builder.assert_bools(local.t1_low_bits.clone());
        builder.assert_bool(local.s1_carry.clone());

        // Pack x, square, and helper limbs.
        let x0: AB::Expr = pack_bits_le(local.x_bits[..LIMB_BITS].iter().cloned());
        let x1: AB::Expr = pack_bits_le(local.x_bits[LIMB_BITS..].iter().cloned());
        let s0: AB::Expr = pack_bits_le(local.sq_bits[..LIMB_BITS].iter().cloned());
        let s1: AB::Expr = pack_bits_le(local.sq_bits[LIMB_BITS..].iter().cloned());
        let x0_sq_hi: AB::Expr = pack_bits_le(local.x0_sq_hi_bits.iter().cloned());
        let t1_low: AB::Expr = pack_bits_le(local.t1_low_bits.iter().cloned());

        let two_pow_16 = AB::Expr::from_u64(1u64 << LIMB_BITS);

        // Enforce the 32-bit square decomposition using 16-bit limbs.
        // t0 = x0^2 = s0 + 2^16 * x0_sq_hi
        let t0 = x0.clone() * x0.clone();
        builder.assert_eq(t0, s0.clone() + two_pow_16.clone() * x0_sq_hi.clone());

        // t1 = 2 * x0 * x1 = t1_low + 2^16 * t1_hi
        let t1 = x0.clone() * x1.clone() * AB::Expr::TWO;
        builder.assert_eq(
            t1,
            t1_low.clone() + two_pow_16.clone() * local.t1_hi.clone().into(),
        );

        // s1 = (x0_sq_hi + t1_low) mod 2^16
        builder.assert_eq(
            x0_sq_hi + t1_low,
            s1 + two_pow_16.clone() * local.s1_carry.clone().into(),
        );

        // rot_bits is ROTR^3(x_bits).
        for i in 0..NUM_BITS {
            builder.assert_eq(local.rot_bits[i].clone(), local.x_bits[(i + 3) % NUM_BITS].clone());
        }

        // out_bits = sq_bits XOR rot_bits.
        for i in 0..NUM_BITS {
            let xor = local.sq_bits[i]
                .clone()
                .into()
                .xor(&local.rot_bits[i].clone().into());
            builder.assert_eq(local.out_bits[i].clone(), xor);
        }

        // Transition: next.x_bits = local.out_bits.
        let mut when_transition = builder.when_transition();
        for i in 0..NUM_BITS {
            when_transition.assert_eq(local.out_bits[i].clone(), next.x_bits[i].clone());
        }

        // First row input equals public input x0.
        let pis = builder.public_values();
        let x0_low = pis[0].clone();
        let x0_high = pis[1].clone();
        let mut when_first = builder.when_first_row();
        when_first.assert_eq(x0, x0_low);
        when_first.assert_eq(x1, x0_high);
    }
}

#[derive(Clone, Copy, Debug)]
struct IterFRow<F> {
    x_bits: [F; NUM_BITS],
    sq_bits: [F; NUM_BITS],
    rot_bits: [F; NUM_BITS],
    out_bits: [F; NUM_BITS],
    x0_sq_hi_bits: [F; LIMB_BITS],
    t1_low_bits: [F; LIMB_BITS],
    s1_carry: F,
    t1_hi: F,
}

impl<F> Borrow<IterFRow<F>> for [F] {
    fn borrow(&self) -> &IterFRow<F> {
        debug_assert_eq!(self.len(), TRACE_WIDTH);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<IterFRow<F>>() };
        debug_assert!(prefix.is_empty());
        debug_assert!(suffix.is_empty());
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

/// Generate a trace for `2^13` steps, starting from `x0`.
///
/// Each row stores the input bits `x_i`, the square bits, the rotated bits,
/// and the output bits `x_{i+1}` so the final row contains `x_{2^13}` in `out_bits`.
pub fn generate_trace_rows<F: PrimeField64>(x0: u32) -> RowMajorMatrix<F> {
    let n = NUM_STEPS;
    let mut trace = RowMajorMatrix::new(F::zero_vec(n * TRACE_WIDTH), TRACE_WIDTH);

    let (prefix, rows, suffix) = unsafe { trace.values.align_to_mut::<IterFRow<F>>() };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(rows.len(), n);

    let mut x = x0;
    for row in rows.iter_mut() {
        let sq = x.wrapping_mul(x);
        let rot = x.rotate_right(3);
        let out = sq ^ rot;

        let x_bits = bits_le_32::<F>(x);
        let sq_bits = bits_le_32::<F>(sq);
        let rot_bits = bits_le_32::<F>(rot);
        let out_bits = bits_le_32::<F>(out);

        let x0_low = (x & 0xffff) as u64;
        let x1_high = (x >> 16) as u64;
        let t0 = x0_low * x0_low;
        let x0_sq_hi = ((t0 >> 16) & 0xffff) as u16;

        let t1 = 2 * x0_low * x1_high;
        let t1_low = (t1 & 0xffff) as u16;
        let t1_hi = (t1 >> 16) as u64;

        let s1_carry = ((x0_sq_hi as u64 + t1_low as u64) >> 16) & 1;

        row.x_bits = x_bits;
        row.sq_bits = sq_bits;
        row.rot_bits = rot_bits;
        row.out_bits = out_bits;
        row.x0_sq_hi_bits = bits_le_16::<F>(x0_sq_hi);
        row.t1_low_bits = bits_le_16::<F>(t1_low);
        row.s1_carry = F::from_u64(s1_carry);
        row.t1_hi = F::from_u64(t1_hi);

        x = out;
    }

    trace
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
