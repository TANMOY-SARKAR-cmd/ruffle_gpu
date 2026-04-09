import fs from "fs/promises";
import path from "path";
import url from "url";
import archiver from "archiver";

async function zip(source: string, destination: string) {
    const resolvedSource = path.resolve(source);
    const resolvedDestination = path.resolve(destination);
    if (!resolvedDestination.endsWith(".xpi") && !resolvedDestination.endsWith(".zip")) {
        throw new Error("Destination must be an .xpi or .zip file.");
    }
    const relDestination = path.relative(process.cwd(), resolvedDestination);
    if (
        path.isAbsolute(relDestination) ||
        relDestination === ".." ||
        relDestination.startsWith(".." + path.sep)
    ) {
        throw new Error(
            "Destination must be within the current working directory.",
        );
    }
    await fs.mkdir(path.dirname(resolvedDestination), { recursive: true });
    const output = (await fs.open(resolvedDestination, "w")).createWriteStream();
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

    archive.directory(resolvedSource, "");

    await archive.finalize();
}
const assets = url.fileURLToPath(new URL("../assets/", import.meta.url));
zip(assets, process.argv[2] ?? "").catch(console.error);
