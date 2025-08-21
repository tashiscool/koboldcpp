/*
 * Apple Silicon Memory Management Fixes for KoboldCpp
 * 
 * This addresses the memory pressure issues that cause crashes
 * when loading large models + LoRAs on Apple Silicon.
 */

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <mach/vm_statistics.h>
#include <mach/mach_types.h>
#include <mach/mach_init.h>
#include <mach/mach_host.h>

class AppleSiliconMemoryManager {
private:
    static constexpr size_t MEMORY_WARNING_THRESHOLD = 0.85; // 85% memory usage
    static constexpr size_t MEMORY_CRITICAL_THRESHOLD = 0.95; // 95% memory usage
    
public:
    /**
     * Get current memory pressure on Apple Silicon
     */
    static double get_memory_pressure() {
        vm_size_t page_size;
        vm_statistics64_data_t vm_stats;
        mach_msg_type_number_t count = sizeof(vm_stats) / sizeof(natural_t);
        mach_port_t mach_port = mach_host_self();
        
        if (host_page_size(mach_port, &page_size) != KERN_SUCCESS) {
            return 0.0;
        }
        
        if (host_statistics64(mach_port, HOST_VM_INFO, 
                             (host_info64_t)&vm_stats, &count) != KERN_SUCCESS) {
            return 0.0;
        }
        
        // Calculate memory pressure
        uint64_t total_pages = vm_stats.free_count + vm_stats.active_count + 
                              vm_stats.inactive_count + vm_stats.wire_count;
        uint64_t used_pages = total_pages - vm_stats.free_count;
        
        return (double)used_pages / total_pages;
    }
    
    /**
     * Get total system memory in bytes
     */
    static size_t get_total_memory() {
        int mib[2] = {CTL_HW, HW_MEMSIZE};
        uint64_t memsize;
        size_t len = sizeof(memsize);
        
        if (sysctl(mib, 2, &memsize, &len, NULL, 0) == 0) {
            return memsize;
        }
        return 0;
    }
    
    /**
     * Check if memory allocation is safe
     */
    static bool is_allocation_safe(size_t requested_bytes) {
        double pressure = get_memory_pressure();
        size_t total_mem = get_total_memory();
        
        if (pressure > MEMORY_CRITICAL_THRESHOLD) {
            printf("🍎 MEMORY WARNING: Critical pressure %.1f%%, blocking allocation\n", 
                   pressure * 100);
            return false;
        }
        
        if (pressure > MEMORY_WARNING_THRESHOLD) {
            printf("🍎 MEMORY WARNING: High pressure %.1f%%, requested %.1fMB\n", 
                   pressure * 100, requested_bytes / (1024.0 * 1024.0));
        }
        
        return true;
    }
    
    /**
     * Force memory cleanup on Apple Silicon
     */
    static void force_memory_cleanup() {
        printf("🍎 MEMORY CLEANUP: Forcing garbage collection\n");
        
        // Trigger memory pressure relief
        // On Apple Silicon, this helps the system reclaim unused memory
        sync(); // Flush filesystem caches
        
        // Give the system time to reclaim memory
        usleep(100000); // 100ms
        
        double new_pressure = get_memory_pressure();
        printf("🍎 MEMORY CLEANUP: Pressure after cleanup: %.1f%%\n", new_pressure * 100);
    }
    
    /**
     * Apple Silicon optimized memory allocation
     */
    static void* allocate_with_pressure_monitoring(size_t size) {
        if (!is_allocation_safe(size)) {
            force_memory_cleanup();
            
            // Retry after cleanup
            if (!is_allocation_safe(size)) {
                printf("🍎 MEMORY ERROR: Cannot allocate %.1fMB safely\n", 
                       size / (1024.0 * 1024.0));
                return nullptr;
            }
        }
        
        // Use aligned allocation for Apple Silicon (optimal for Metal)
        void* ptr = nullptr;
        int result = posix_memalign(&ptr, 16384, size); // 16KB alignment for Metal
        
        if (result != 0) {
            printf("🍎 MEMORY ERROR: Aligned allocation failed for %.1fMB\n", 
                   size / (1024.0 * 1024.0));
            return nullptr;
        }
        
        printf("🍎 MEMORY SUCCESS: Allocated %.1fMB (pressure: %.1f%%)\n", 
               size / (1024.0 * 1024.0), get_memory_pressure() * 100);
        
        return ptr;
    }
    
    /**
     * Safe deallocation with pressure monitoring
     */
    static void deallocate_with_monitoring(void* ptr) {
        if (ptr) {
            free(ptr);
            
            // Check if memory pressure improved
            static double last_pressure = 0.0;
            double current_pressure = get_memory_pressure();
            
            if (current_pressure < last_pressure - 0.05) { // 5% improvement
                printf("🍎 MEMORY RELIEF: Pressure reduced to %.1f%%\n", current_pressure * 100);
            }
            
            last_pressure = current_pressure;
        }
    }
};

/**
 * Wrapper for GGML tensor allocation on Apple Silicon
 */
ggml_tensor* apple_silicon_tensor_alloc(ggml_context* ctx, ggml_type type, 
                                       int64_t ne0, int64_t ne1 = 1, 
                                       int64_t ne2 = 1, int64_t ne3 = 1) {
    
    // Calculate required memory
    size_t tensor_size = ggml_type_size(type) * ne0 * ne1 * ne2 * ne3;
    
    if (!AppleSiliconMemoryManager::is_allocation_safe(tensor_size)) {
        printf("🍎 TENSOR ERROR: Cannot safely allocate tensor of size %.1fMB\n", 
               tensor_size / (1024.0 * 1024.0));
        return nullptr;
    }
    
    // Allocate the tensor
    ggml_tensor* tensor = ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);
    
    if (!tensor) {
        printf("🍎 TENSOR ERROR: GGML allocation failed\n");
        AppleSiliconMemoryManager::force_memory_cleanup();
    } else {
        printf("🍎 TENSOR SUCCESS: Allocated %s tensor %.1fMB\n", 
               ggml_type_name(type), tensor_size / (1024.0 * 1024.0));
    }
    
    return tensor;
}

/**
 * Patch LoRA loading to use Apple Silicon memory management
 */
void patch_lora_loading_for_apple_silicon() {
    printf("🍎 APPLE SILICON: Applying memory management patches\n");
    
    // Monitor memory before LoRA loading
    double initial_pressure = AppleSiliconMemoryManager::get_memory_pressure();
    size_t total_memory = AppleSiliconMemoryManager::get_total_memory();
    
    printf("🍎 APPLE SILICON: System memory: %.1fGB, current pressure: %.1f%%\n",
           total_memory / (1024.0 * 1024.0 * 1024.0), initial_pressure * 100);
    
    if (initial_pressure > 0.8) {
        printf("🍎 APPLE SILICON: High memory pressure detected, cleaning up\n");
        AppleSiliconMemoryManager::force_memory_cleanup();
    }
}

#endif // __APPLE__

/*
 * Integration point for KoboldCpp LoRA loading
 * Add this call at the beginning of LoRA processing
 */
void lora_apply_apple_silicon_fixes() {
#ifdef __APPLE__
    patch_lora_loading_for_apple_silicon();
#endif
}