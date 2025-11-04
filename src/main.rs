use catapult::numerics::VectorLike;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    const DIM: usize = 128;

    // Initialize with some random-ish values
    let vec1: Vec<f32> = (0..DIM).map(|i| (i as f32 * 0.1).sin()).collect();
    let vec2: Vec<f32> = (0..DIM).map(|i| (i as f32 * 0.2).cos()).collect();

    // Warm-up phase
    for _ in 0..1000 {
        black_box(vec1.l2_squared(&vec2));
    }

    // Main benchmark loop - call l2_squared many times
    // This should be enough iterations for cargo flame to get a good profile
    println!("Starting benchmark...");
    let iterations = 1_000_000_000;

    let start = Instant::now();
    let mut sum = 0.0f32;
    for i in 0..iterations {
        let result = black_box(vec1.l2_squared(black_box(&vec2)));
        sum += result;

        if i % 10_000_000 == 0 && i > 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = i as f64 / elapsed;
            println!(
                "Progress: {}/{} ({:.1}M calls/sec)",
                i,
                iterations,
                rate / 1_000_000.0
            );
        }
    }
    let elapsed = start.elapsed();

    // Print results
    println!("\n=== RESULTS WITH ASSERTS ===");
    println!("Completed {} iterations", iterations);
    println!("Total time: {:.3}s", elapsed.as_secs_f64());
    println!(
        "Time per call: {:.2}ns",
        elapsed.as_nanos() as f64 / iterations as f64
    );
    println!(
        "Calls per second: {:.2}M/s",
        iterations as f64 / elapsed.as_secs_f64() / 1_000_000.0
    );
    println!("Sum: {}", black_box(sum));

    println!("\n=== TO MEASURE ASSERT OVERHEAD ===");
    println!(
        "1. Note the 'Time per call' above: {:.2}ns",
        elapsed.as_nanos() as f64 / iterations as f64
    );
    println!("2. Edit src/numerics/f32slice.rs:55-56");
    println!("   Change: assert!(self.len() == othr.len());");
    println!("   To:     debug_assert!(self.len() == othr.len());");
    println!("   And:    assert!(self.len().is_multiple_of(SIMD_LANECOUNT));");
    println!("   To:     debug_assert!(self.len().is_multiple_of(SIMD_LANECOUNT));");
    println!("3. Run 'cargo run --release' again");
    println!("4. Compare times - the difference is the assert overhead");
}
