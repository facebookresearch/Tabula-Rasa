/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * WGSL Preprocessor - Adds #include support for modular WGSL shaders
 *
 * Usage:
 *   const preprocessor = new WGSLPreprocessor('shaders/');
 *   const code = await preprocessor.process('shaders/main.wgsl');
 *
 * In WGSL files:
 *   #include "common.wgsl"
 *   #include "prng/threefry.wgsl"
 */

class WGSLPreprocessor {
    constructor(basePath = '') {
        this.basePath = basePath;
        this.cache = new Map();
        this.includeStack = new Set(); // For circular dependency detection
    }

    /**
     * Resolve a path relative to a base file
     */
    resolvePath(includePath, currentFile) {
        // If includePath starts with '/', treat as absolute from basePath
        if (includePath.startsWith('/')) {
            return this.basePath + includePath.slice(1);
        }

        // Get directory of current file
        const currentDir = currentFile.substring(0, currentFile.lastIndexOf('/') + 1);

        // Handle relative paths
        if (includePath.startsWith('./')) {
            return currentDir + includePath.slice(2);
        }

        if (includePath.startsWith('../')) {
            let dir = currentDir;
            let path = includePath;
            while (path.startsWith('../')) {
                dir = dir.substring(0, dir.slice(0, -1).lastIndexOf('/') + 1);
                path = path.slice(3);
            }
            return dir + path;
        }

        // Default: relative to current file's directory
        return currentDir + includePath;
    }

    /**
     * Fetch a file (with caching)
     */
    async fetchFile(path) {
        if (this.cache.has(path)) {
            return this.cache.get(path);
        }

        // Add cache-busting timestamp to ensure fresh file
        const response = await fetch(path);
        if (!response.ok) {
            throw new Error(`Failed to load WGSL file: ${path} (${response.status})`);
        }

        const content = await response.text();
        this.cache.set(path, content);
        return content;
    }

    /**
     * Process a WGSL file, resolving all #include directives
     */
    async process(filePath) {
        // Check for circular includes
        if (this.includeStack.has(filePath)) {
            const cycle = [...this.includeStack, filePath].join(' -> ');
            throw new Error(`Circular include detected: ${cycle}`);
        }

        this.includeStack.add(filePath);

        try {
            let code = await this.fetchFile(filePath);

            // Match #include "path" or #include <path>
            const includeRegex = /^[ \t]*#include\s+["<]([^">]+)[">][ \t]*$/gm;

            // Find all includes
            const includes = [];
            let match;
            while ((match = includeRegex.exec(code)) !== null) {
                includes.push({
                    fullMatch: match[0],
                    path: match[1],
                    index: match.index
                });
            }

            // Process includes in reverse order to preserve indices
            for (let i = includes.length - 1; i >= 0; i--) {
                const inc = includes[i];
                const resolvedPath = this.resolvePath(inc.path, filePath);

                // Recursively process included file
                const includedCode = await this.process(resolvedPath);

                // Add source comment for debugging
                const wrappedCode = [
                    `// ========== BEGIN: ${inc.path} ==========`,
                    includedCode.trim(),
                    `// ========== END: ${inc.path} ==========`,
                    ''
                ].join('\n');

                // Replace the #include directive
                code = code.slice(0, inc.index) + wrappedCode + code.slice(inc.index + inc.fullMatch.length);
            }

            return code;
        } finally {
            this.includeStack.delete(filePath);
        }
    }

    /**
     * Process a WGSL string (for inline shader code)
     */
    async processString(code, virtualPath = 'inline.wgsl') {
        // Temporarily store in cache
        this.cache.set(virtualPath, code);
        const result = await this.process(virtualPath);
        this.cache.delete(virtualPath);
        return result;
    }

    /**
     * Clear the file cache
     */
    clearCache() {
        this.cache.clear();
    }

    /**
     * Preload multiple files into cache
     */
    async preload(paths) {
        await Promise.all(paths.map(p => this.fetchFile(p)));
    }
}

// Also support #pragma once for include guards
class WGSLPreprocessorWithGuards extends WGSLPreprocessor {
    constructor(basePath = '') {
        super(basePath);
        this.included = new Set();
    }

    async process(filePath) {
        // Skip if already included (implements #pragma once behavior)
        const normalizedPath = filePath.toLowerCase();
        if (this.included.has(normalizedPath)) {
            return ''; // Already included
        }

        let code = await this.fetchFile(filePath);

        // Check for #pragma once
        const pragmaOnceRegex = /^[ \t]*#pragma\s+once[ \t]*$/m;
        if (pragmaOnceRegex.test(code)) {
            this.included.add(normalizedPath);
            code = code.replace(pragmaOnceRegex, '// #pragma once (processed)');
        }

        // Check for circular includes (only if not using #pragma once)
        if (!this.included.has(normalizedPath) && this.includeStack.has(filePath)) {
            const cycle = [...this.includeStack, filePath].join(' -> ');
            throw new Error(`Circular include detected: ${cycle}`);
        }

        this.includeStack.add(filePath);

        try {
            // Match #include "path" or #include <path>
            const includeRegex = /^[ \t]*#include\s+["<]([^">]+)[">][ \t]*$/gm;

            const includes = [];
            let match;
            while ((match = includeRegex.exec(code)) !== null) {
                includes.push({
                    fullMatch: match[0],
                    path: match[1],
                    index: match.index
                });
            }

            // Process includes in reverse order
            for (let i = includes.length - 1; i >= 0; i--) {
                const inc = includes[i];
                const resolvedPath = this.resolvePath(inc.path, filePath);

                const includedCode = await this.process(resolvedPath);

                let wrappedCode = '';
                if (includedCode.trim()) {
                    wrappedCode = [
                        `// ========== BEGIN: ${inc.path} ==========`,
                        includedCode.trim(),
                        `// ========== END: ${inc.path} ==========`,
                        ''
                    ].join('\n');
                }

                code = code.slice(0, inc.index) + wrappedCode + code.slice(inc.index + inc.fullMatch.length);
            }

            return code;
        } finally {
            this.includeStack.delete(filePath);
        }
    }

    /**
     * Reset include guards (call before processing a new entry point)
     */
    resetGuards() {
        this.included.clear();
        this.cache.clear();  // Also clear file cache to ensure fresh files
    }
}

// Export for ES modules (if supported) or attach to window
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { WGSLPreprocessor, WGSLPreprocessorWithGuards };
} else if (typeof window !== 'undefined') {
    window.WGSLPreprocessor = WGSLPreprocessor;
    window.WGSLPreprocessorWithGuards = WGSLPreprocessorWithGuards;
}
