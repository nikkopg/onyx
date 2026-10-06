// Prepares the source tree for a web build with no Enterprise Edition code.
// Deletes the ee directories and maps their imports to src/community-stubs.
// Run it only in a throwaway build tree (the Docker builder stage): it edits
// tsconfig.json and deletes files.
import { readFileSync, rmSync, writeFileSync } from "node:fs";

const EE_DIRS = ["src/ee", "src/app/ee"];
const STUB_PATHS = {
  "@/ee/*": ["./src/community-stubs/enterprise/*"],
  "@/app/ee/*": ["./src/community-stubs/app-enterprise/*"],
};

for (const dir of EE_DIRS) {
  rmSync(dir, { recursive: true, force: true });
}

const tsconfig = JSON.parse(readFileSync("tsconfig.json", "utf8"));
tsconfig.compilerOptions.paths = {
  ...STUB_PATHS,
  ...tsconfig.compilerOptions.paths,
};
writeFileSync("tsconfig.json", `${JSON.stringify(tsconfig, null, 2)}\n`);

console.log(
  `Community build: removed ${EE_DIRS.join(", ")}; mapped ${Object.keys(STUB_PATHS).join(", ")} to src/community-stubs.`
);
