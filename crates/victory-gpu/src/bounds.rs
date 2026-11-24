//! Bounds on the work a title command asks the host to do.
//!
//! The title's own counts reach the host GPU as they are: a dispatch's
//! group counts, a draw's vertex and instance counts, a mesh dispatch's
//! groups. When the console's code runs on the host these are the values the
//! title computed, so they are usually right. They are not always: a pass
//! that VIctory dropped or translated wrongly leaves whatever its output
//! buffer held, and a count read from there can be enormous. The host
//! answers a dispatch of four billion groups or a draw of four billion
//! vertices by running until its watchdog resets the device, which ends the
//! run (see `host::fail`).
//!
//! The indirect path already bounds its arguments on the GPU
//! ([`crate::indirect`]). The direct path needs the same protection, and it
//! needs it on the CPU: the count is already known when the command is
//! recorded, so there is no reason to let it reach the GPU unchecked.
//!
//! The bounds are the console's own limits, as the indirect path uses them:
//! 2^24 vertices, 2^26 vertex instances, 65,535 groups per dimension, and
//! 2^20 groups for a mesh dispatch. A count within them is the title's
//! business; one beyond them is a value no legitimate title command has, so
//! clamping it costs nothing real and saves the run.

/// The most vertices one draw may ask for.
pub(crate) const MAX_VERTICES: u32 = 1 << 24;

/// The most vertex instances one draw may ask for.
pub(crate) const MAX_INSTANCES: u32 = 1 << 26;

/// The most groups one dimension of a dispatch may ask for.
pub(crate) const MAX_GROUPS: u32 = 1 << 16;

/// The most groups one dispatch may ask for in total, so that
/// `x * y * z` (which does not fit in 32 bits for large counts) stays
/// inside the host's per-dispatch limit.
pub(crate) const MAX_DISPATCH_GROUPS: u64 = 1 << 20;

/// How many powers of two to take off every work bound (diagnostics). A
/// translated shader can spend hundreds of calls per thread; a title's
/// original group counts then put more work in one list than the host can
/// finish before its watchdog resets the device. Running a bounded fraction
/// of the work keeps the run alive and still draws real geometry.
pub(crate) fn work_shift() -> u32 {
    #[cfg(feature = "diagnostics")]
    {
        std::env::var("VICTORY_WORK_SHIFT")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(0)
            .min(20)
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        0
    }
}

fn capped(value: u32) -> u32 {
    value >> work_shift()
}

/// The group counts of a dispatch, bounded.
pub(crate) fn dispatch_groups(x: u32, y: u32, z: u32) -> (u32, u32, u32) {
    let cap = capped(MAX_GROUPS);
    let (x, y, z) = (x.min(cap), y.min(cap), z.min(cap));
    // The product of the counts is the work; keep it inside the limit.
    if u64::from(x) * u64::from(y) * u64::from(z) > u64::from(capped(MAX_DISPATCH_GROUPS as u32)) {
        return (0, 0, 0);
    }
    (x, y, z)
}

/// The vertex and instance counts of a draw, bounded.
pub(crate) fn draw_counts(vertices: u32, instances: u32) -> (u32, u32) {
    let vertices = vertices.min(capped(MAX_VERTICES));
    // The vertices of every instance run, so the product is the work.
    let instances = instances.min(capped(MAX_INSTANCES) / vertices.max(1));
    (vertices, instances)
}

/// The most groups one mesh dispatch may ask for, bounded.
pub(crate) fn mesh_groups(x: u32) -> u32 {
    x.min(capped(MAX_GROUPS))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_counts_within_the_limits_are_kept() {
        assert_eq!(dispatch_groups(8, 8, 1), (8, 8, 1));
        assert_eq!(dispatch_groups(64, 1, 1), (64, 1, 1));
    }

    #[test]
    fn dispatch_counts_beyond_them_are_cut() {
        assert_eq!(dispatch_groups(u32::MAX, 1, 1), (MAX_GROUPS, 1, 1));
        assert_eq!(dispatch_groups(65_536, 1, 1), (65_536, 1, 1));
        assert_eq!(dispatch_groups(1 << 17, 1, 1), (MAX_GROUPS, 1, 1));
    }

    #[test]
    fn a_dispatch_of_a_huge_total_runs_empty() {
        assert_eq!(dispatch_groups(65_535, 65_535, 1), (0, 0, 0));
        assert_eq!(dispatch_groups(u32::MAX, u32::MAX, u32::MAX), (0, 0, 0));
    }

    #[test]
    fn draw_counts_within_the_limits_are_kept() {
        assert_eq!(draw_counts(100, 2), (100, 2));
        // No vertices means no work, so the instances stay as they are.
        assert_eq!(draw_counts(0, 10), (0, 10));
    }

    #[test]
    fn draw_counts_beyond_them_are_cut() {
        assert_eq!(draw_counts(u32::MAX, 1), (MAX_VERTICES, 1));
        assert_eq!(draw_counts(1, u32::MAX), (1, MAX_INSTANCES));
    }

    #[test]
    fn the_vertices_of_every_instance_stay_inside_the_limit() {
        let (vertices, instances) = draw_counts(MAX_VERTICES, MAX_INSTANCES);
        assert!(u64::from(vertices) * u64::from(instances) <= u64::from(MAX_INSTANCES));
    }
}
