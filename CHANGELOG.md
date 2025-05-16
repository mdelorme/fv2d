# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1] - 2025-05-16

### Added
- Gresho Vortex test case for Low-Mach flows
- List of contributors on the main page
- This Changelog

### Removed
- Spurious `KOKKOS_INLINE_FUNCTION` before `computeSlopes`

### Changed
- Separater `DeviceParams` and `Params` to avoid copying strings on GPU and generating a lot of warnings
- Updated minimum version of C++ to 20
- Updated HighFive version to account for root attribute reading problems in restarts