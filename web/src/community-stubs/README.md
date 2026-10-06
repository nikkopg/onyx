# Community stubs

MIT stand-ins for the Enterprise modules that Community Edition code imports.

`scripts/community-build.mjs` deletes `src/ee` and `src/app/ee` and maps the
`@/ee/*` and `@/app/ee/*` imports to `enterprise/` and `app-enterprise/` here.
The default build does not use these files.

Each stub does what the Community Edition does today with the Enterprise
code turned off. Keep the exported names and types the same as the modules
they replace; the community build fails if they differ.
