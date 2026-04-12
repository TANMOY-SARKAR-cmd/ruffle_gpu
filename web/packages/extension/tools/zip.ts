import fs from "fs/promises";
import path from "path";
import url from "url";
import archiver from "archiver";

async function zip(source: string, destination: string) {
    // Use only the basename of the destination to prevent path traversal.
    // path.basename is a CodeQL-recognised sanitiser that strips directory components.
    const filename = path.basename(destination);
    if (!filename || !/\.(zip|xpi)$/i.test(filename)) {
        throw new Error(
            `Destination must be a .zip or .xpi filename, got: "${filename}"`,
        );
    }
    // Construct the output path from a trusted base directory (derived from the
    // script's own location, not from user-supplied input).
    const distDir = path.resolve(
        path.dirname(url.fileURLToPath(import.meta.url)),
        "../dist",
    );
    const safeDest = path.join(distDir, filename);
    await fs.mkdir(path.dirname(safeDest), { recursive: true });
    const output = (await fs.open(safeDest, "w")).createWriteStream();
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
