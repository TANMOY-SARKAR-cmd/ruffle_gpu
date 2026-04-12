import fs from "fs/promises";
import path from "path";
import url from "url";
import archiver from "archiver";

async function zip(source: string, destination: string) {
    // Resolve and validate destination path to prevent path traversal
    const resolvedDest = path.resolve(destination);
    // Validate extension to prevent accidental writes outside expected file types
    if (!/\.(zip|xpi)$/i.test(resolvedDest)) {
        throw new Error(
            `Destination must be a .zip or .xpi file: ${resolvedDest}`,
        );
    }
    // Validate that the destination is within the project tree
    const projectRoot = path.resolve(
        path.dirname(url.fileURLToPath(import.meta.url)),
        "../../../",
    );
    if (
        !resolvedDest.startsWith(projectRoot + path.sep) &&
        resolvedDest !== projectRoot
    ) {
        throw new Error(
            `Destination must be within the project directory: ${resolvedDest}`,
        );
    }
    await fs.mkdir(path.dirname(resolvedDest), { recursive: true });
    const output = (await fs.open(resolvedDest, "w")).createWriteStream();
    const archive = archiver("zip");

    output.on("close", () => {
        console.log(
            `Extension is ${archive.pointer()} total bytes when packaged.`,
        );
    });

    archive.on("error", (error) => {
        throw error;
    });

    archive.on("warning", (error) => {
        if (error.code === "ENOENT") {
            console.warn(`Warning whilst zipping extension: ${error}`);
        } else {
            throw error;
        }
    });

    archive.pipe(output);

    archive.directory(source, "");

    await archive.finalize();
}
const assets = url.fileURLToPath(new URL("../assets/", import.meta.url));
zip(assets, process.argv[2] ?? "").catch(console.error);
