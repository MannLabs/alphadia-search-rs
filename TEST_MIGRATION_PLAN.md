# TEST MIGRATION PLAN: Move Rust Tests from `src/` to `tests/`

## General Instructions (Strict)

- **Do not add any additional tests.**
- **Do not change any functionality.**
- **Do not reformat code.**
- **Only move existing test code as described in this plan.**
- **Each PR should be as small and focused as possible (ideally one module per PR).**

---

## Acceptance Criteria

- After each migration, **all tests must be run using `cargo test` and all must pass**.
- PRs that do not pass all tests will not be accepted.

---

## General Steps for Each Module

1. **Create a new test file in `/tests`** named after the module (e.g., `tests/rt_index.rs`).
2. **Move all test functions and helpers** from the `#[cfg(test)] mod tests` in the source file to the new test file.
3. **Update imports** in the new test file to use the public API of the crate (e.g., `use alpha_rs::rt_index::*;`).
4. **Remove the old test code** from the source file.
5. **Run `cargo test`** to ensure tests are discovered and pass.
6. **Repeat for each module in a separate PR.**

---

## Module-by-Module Instructions

### 1. `benchmark.rs`

- **Tests found:** `mod tests` at the end of the file, with `test_convolution_similarity`.
- **Action:**
  - Create `tests/benchmark.rs`.
  - Move the entire `mod tests` (including `test_convolution_similarity`) to `tests/benchmark.rs`.
  - Update imports to use the public API.
  - Remove the `mod tests` from `benchmark.rs`.

---

### 2. `convolution.rs` and `convolution_test.rs`

- **Tests found:** `mod tests` in `convolution.rs` (empty), and all actual tests in `convolution_test.rs`.
- **Action:**
  - Create `tests/convolution_test.rs`.
  - Move all test functions from `src/convolution_test.rs` to `tests/convolution_test.rs`.
  - Update imports to use the public API.
  - Remove `src/convolution_test.rs` and the `mod convolution_test;` reference in `lib.rs` if present.

---

### 3. `peak_group_scoring.rs`

- **Tests found:** `mod tests` with multiple test functions.
- **Action:**
  - Create `tests/peak_group_scoring.rs`.
  - Move all test functions and helpers from the `mod tests` to the new file.
  - Update imports to use the public API.
  - Remove the `mod tests` from `peak_group_scoring.rs`.

---

### 4. `rt_index.rs`

- **Tests found:** `mod tests` with multiple test functions and a helper.
- **Action:**
  - Create `tests/rt_index.rs`.
  - Move all test functions and helpers from the `mod tests` to the new file.
  - Update imports to use the public API.
  - Remove the `mod tests` from `rt_index.rs`.

---

### 5. `mz_index.rs`

- **Tests found:** `mod tests` with submodules and multiple test functions.
- **Action:**
  - Create `tests/mz_index.rs`.
  - Move all test functions and submodules from the `mod tests` to the new file.
  - Update imports to use the public API.
  - Remove the `mod tests` from `mz_index.rs`.

---

### 6. `quadrupole_observation.rs`

- **Tests found:** `mod tests` with multiple test functions.
- **Action:**
  - Create `tests/quadrupole_observation.rs`.
  - Move all test functions and helpers from the `mod tests` to the new file.
  - Update imports to use the public API.
  - Remove the `mod tests` from `quadrupole_observation.rs`.

---

### 7. `xic_slice.rs`

- **Tests found:** `mod tests` (currently empty).
- **Action:**
  - Create `tests/xic_slice.rs`.
  - Move the (empty) `mod tests` to the new file for consistency.
  - Remove the `mod tests` from `xic_slice.rs`.

---

### 8. `dia_data_builder.rs`

- **Tests found:** `mod tests` with test functions and a helper.
- **Action:**
  - Create `tests/dia_data_builder.rs`.
  - Move all test functions and helpers from the `mod tests` to the new file.
  - Update imports to use the public API.
  - Remove the `mod tests` from `dia_data_builder.rs`.

---

### 9. `kernel.rs`

- **Tests found:** `mod tests` with test functions.
- **Action:**
  - Create `tests/kernel.rs`.
  - Move all test functions from the `mod tests` to the new file.
  - Update imports to use the public API.
  - Remove the `mod tests` from `kernel.rs`.

---

### 10. `candidate.rs`

- **Tests found:** None.
- **Action:** No migration needed.

---

## Special Notes

- If a test needs access to private items, consider making them `pub(crate)` or re-exporting for test purposes.
- If any test helpers are used across multiple modules, consider creating a `tests/common.rs` for shared code.
- After each migration, run `cargo test` to ensure all tests are still discovered and passing.

---

**Repeat the above steps for each module in a separate PR for clarity and easy review.**