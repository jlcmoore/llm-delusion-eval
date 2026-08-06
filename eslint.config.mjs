import js from "@eslint/js";
import globals from "globals";
import prettierConfig from "eslint-config-prettier";
import prettierPlugin from "eslint-plugin-prettier";
import sonarjsPlugin from "eslint-plugin-sonarjs";

export default [
  // Base recommended rules for all JS
  js.configs.recommended,

  // Disable rules that conflict with Prettier formatting
  prettierConfig,

  // Surface Prettier formatting issues through ESLint
  {
    plugins: {
      prettier: prettierPlugin,
      sonarjs: sonarjsPlugin,
    },
    rules: {
      "prettier/prettier": "error",
      "sonarjs/cognitive-complexity": ["warn", 15],
      "sonarjs/no-all-duplicated-branches": "warn",
      "sonarjs/no-identical-functions": "warn",
      "sonarjs/no-duplicate-string": ["warn", { threshold: 3 }],
    },
  },

  // Global ignores
  {
    ignores: [
      "**/node_modules/**",
      "**/dist/**",
      "**/.venv/**",
    ],
  },

  // plain browser JS
  {
    files: ["src/llm_delusion_eval/scripts/report_assets/**/*.js"],
    languageOptions: {
      globals: {
        ...globals.browser,
        Plotly: "readonly",
      },
    },
    rules: {
      "no-unused-vars": ["error", { varsIgnorePattern: "^(closeModal|modelItems|codesInCat)$", argsIgnorePattern: "^(modelItems|codesInCat)$" }],
    },
  },
];
