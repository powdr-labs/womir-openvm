#[cfg(feature = "cuda")]
use openvm_cuda_builder::{CudaBuilder, cuda_available};

fn main() {
    #[cfg(feature = "cuda")]
    {
        if !cuda_available() {
            return; // Skip CUDA compilation
        }

        // Get the openvm-circuit crate location from the DEP_ variable set by our dependency
        // chain. Since we use cargo patch to point to local openvm, this gives us the path.
        let circuit_primitives_include =
            std::env::var("DEP_CIRCUIT_PRIMITIVES_CUDA_INCLUDE").unwrap_or_default();

        // The vm CUDA include is at the same level as circuit-primitives
        // primitives is at openvm/crates/circuits/primitives/cuda/include
        // vm is at openvm/crates/vm/cuda/include
        let vm_include = if !circuit_primitives_include.is_empty() {
            let p = std::path::PathBuf::from(&circuit_primitives_include);
            // Go up from primitives/cuda/include to openvm root, then down to vm/cuda/include
            let openvm_root = p
                .parent()
                .and_then(|p| p.parent())
                .and_then(|p| p.parent())
                .and_then(|p| p.parent())
                .unwrap_or(&p);
            openvm_root.join("vm").join("cuda").join("include")
        } else {
            std::path::PathBuf::new()
        };

        let mut builder: CudaBuilder = CudaBuilder::new()
            .include_from_dep("DEP_CUDA_COMMON_INCLUDE")
            .include("cuda/include");

        // Add include paths from dependencies
        if !circuit_primitives_include.is_empty() {
            builder = builder.include(&circuit_primitives_include);
        }
        if vm_include.exists() {
            builder = builder.include(vm_include.to_str().unwrap());
        }

        builder = builder
            .watch("cuda")
            .library_name("tracegen_gpu_womir")
            .files_from_glob("cuda/src/**/*.cu");

        builder.emit_link_directives();
        builder.build();
    }
}
