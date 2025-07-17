# TEST MIGRATION PLAN: Move Rust Tests to Separate Files (Following Best Practices)

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

## PR Message Format

Each migration PR should follow this format:

**Title:** `migrate: Move [module_name] tests to separate file (Step X/9)`

**Description:**
This PR implements step X of the test migration plan, moving [module_name] tests from inline `#[cfg(test)] mod tests` to `src/[module_name]/tests.rs`.

**Changes:**
- Created `src/[module_name]/` directory with `mod.rs` and `tests.rs`
- Moved `src/[module_name].rs` implementation to `src/[module_name]/mod.rs`
- Created `src/[module_name]/tests.rs` with migrated test functions
- Updated `src/lib.rs` module declaration (if needed)
- [Any other specific changes]

**Technical Notes:**
[Include any technical explanations for changes made]

**Testing:**
- All tests passing: [X unit tests + Y integration tests]
- Verified with `cargo test`

**Migration Status:** X/9 modules completed

---

## General Steps for Each Module

1. **Create a new directory** `src/[module_name]/` for the module.
2. **Move the main implementation** from `src/[module_name].rs` to `src/[module_name]/mod.rs`.
3. **Create a new test file** `src/[module_name]/tests.rs`.
4. **Move all test functions and helpers** from the `#[cfg(test)] mod tests` in the source file to the new test file.
5. **Update imports** in the new test file to use `super::*` to access the module's items.
6. **Remove the old test code** from the source file (`mod.rs`).
7. **Run `cargo test`** to ensure tests are discovered and pass.
8. **Repeat for each module in a separate PR.**

---

## Module-by-Module Instructions

### 1. `benchmark.rs`

- **Tests found:** `mod tests` at the end of the file, with `test_convolution_similarity`.
- **Action:**
  - Create `src/benchmark/` directory.
  - Move `src/benchmark.rs` to `src/benchmark/mod.rs`.
  - Create `src/benchmark/tests.rs`.
  - Move the entire `mod tests` (including `test_convolution_similarity`) to `src/benchmark/tests.rs`.
  - Update imports to use `super::*`.
  - Remove the `mod tests` from `mod.rs`.

---

### 2. `convolution.rs` and `convolution_test.rs`

- **Tests found:** `mod tests` in `convolution.rs` (empty), and all actual tests in `convolution_test.rs`.
- **Action:**
  - Create `src/convolution/` directory.
  - Move `src/convolution.rs` to `src/convolution/mod.rs`.
  - Create `src/convolution/tests.rs`.
  - Move all test functions from `src/convolution_test.rs` to `src/convolution/tests.rs`.
  - Update imports to use `super::*`.
  - Remove `src/convolution_test.rs` and the `mod convolution_test;` reference in `lib.rs` if present.
  - Remove the empty `mod tests` from `mod.rs`.

---

### 3. `peak_group_scoring.rs`

- **Tests found:** `mod tests` with multiple test functions.
- **Action:**
  - Create `src/peak_group_scoring/` directory.
  - Move `src/peak_group_scoring.rs` to `src/peak_group_scoring/mod.rs`.
  - Create `src/peak_group_scoring/tests.rs`.
  - Move all test functions and helpers from the `mod tests` to the new file.
  - Update imports to use `super::*`.
  - Remove the `mod tests` from `mod.rs`.

---

### 4. `rt_index.rs`

- **Tests found:** `mod tests` with multiple test functions and a helper.
- **Action:**
  - Create `src/rt_index/` directory.
  - Move `src/rt_index.rs` to `src/rt_index/mod.rs`.
  - Create `src/rt_index/tests.rs`.
  - Move all test functions and helpers from the `mod tests` to the new file.
  - Update imports to use `super::*`.
  - Remove the `mod tests` from `mod.rs`.

---

### 5. `mz_index.rs`

- **Tests found:** `mod tests` with submodules and multiple test functions.
- **Action:**
  - Create `src/mz_index/` directory.
  - Move `src/mz_index.rs` to `src/mz_index/mod.rs`.
  - Create `src/mz_index/tests.rs`.
  - Move all test functions and submodules from the `mod tests` to the new file.
  - Update imports to use `super::*`.
  - Remove the `mod tests` from `mod.rs`.

---

### 6. `quadrupole_observation.rs`

- **Tests found:** `mod tests` with multiple test functions.
- **Action:**
  - Create `src/quadrupole_observation/` directory.
  - Move `src/quadrupole_observation.rs` to `src/quadrupole_observation/mod.rs`.
  - Create `src/quadrupole_observation/tests.rs`.
  - Move all test functions and helpers from the `mod tests` to the new file.
  - Update imports to use `super::*`.
  - Remove the `mod tests` from `mod.rs`.

---

### 7. `xic_slice.rs`

- **Tests found:** `mod tests` (currently empty).
- **Action:**
  - Create `src/xic_slice/` directory.
  - Move `src/xic_slice.rs` to `src/xic_slice/mod.rs`.
  - Create `src/xic_slice/tests.rs`.
  - Move the (empty) `mod tests` to the new file for consistency.
  - Remove the `mod tests` from `mod.rs`.

---

### 8. `dia_data_builder.rs`

- **Tests found:** `mod tests` with test functions and a helper.
- **Action:**
  - Create `src/dia_data_builder/` directory.
  - Move `src/dia_data_builder.rs` to `src/dia_data_builder/mod.rs`.
  - Create `src/dia_data_builder/tests.rs`.
  - Move all test functions and helpers from the `mod tests` to the new file.
  - Update imports to use `super::*`.
  - Remove the `mod tests` from `mod.rs`.

---

### 9. `kernel.rs`

- **Tests found:** `mod tests` with test functions.
- **Action:**
  - Create `src/kernel/` directory.
  - Move `src/kernel.rs` to `src/kernel/mod.rs`.
  - Create `src/kernel/tests.rs`.
  - Move all test functions from the `mod tests` to the new file.
  - Update imports to use `super::*`.
  - Remove the `mod tests` from `mod.rs`.

---

### 10. `candidate.rs`

- **Tests found:** None.
- **Action:** No migration needed.

---

## Special Notes

- Tests will have access to all items in the module using `super::*` imports.
- The `#[cfg(test)]` attribute should be applied to the entire `tests.rs` file content or individual test functions.
- If any test helpers are used across multiple modules, consider creating a `src/common/tests.rs` for shared code.
- After each migration, run `cargo test` to ensure all tests are still discovered and passing.
- This approach keeps unit tests close to the code they test while maintaining clean separation.

---

**Repeat the above steps for each module in a separate PR for clarity and easy review.**
