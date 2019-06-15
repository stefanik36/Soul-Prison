package com.stefanik36.soul_prison.data_sets;

import java.io.*;
import java.util.Enumeration;
import java.util.jar.JarFile;
import java.util.zip.ZipEntry;

public class JarExtractor {
    /**
     * @param args
     *            the first arg should be the path to the parent jar file, and
     *            the second should be the directory where child jar files are
     *            extracted.
     */
    public static void main(String[] args) {
        // JarExtractor ex = new JarExtractor(args[0], args[1]);
        // For this demo, args are prefixed.
        JarExtractor ex = new JarExtractor("Soul-Prison.jar ", "data");
        ex.parseJarFile();
    }

    private String parentPath = null;
    private String extractPath = ".";

    /**
     * @param parentPath
     *            the path to the parent jar file
     * @param extractPath
     *            the path to the directory where child files being extracted
     */
    JarExtractor(String parentPath, String extractPath) {
        this.parentPath = parentPath;
        this.extractPath = extractPath;
    }

    /**
     * @param parentJar
     *            the parent jar file
     * @param extractee
     *            the file to be extracted
     * @return the extracted jar file
     * @throws IOException
     */
    private File extractJarFileFromJar(final JarFile parentJar,
                                       final ZipEntry extractee) throws IOException {
        BufferedInputStream is = new BufferedInputStream(parentJar
                .getInputStream(extractee));

        File f = new File(extractPath + File.separator + extractee.getName());
        String parentName = f.getParent();
        if (parentName != null) {
            File dir = new File(parentName);
            dir.mkdirs();
        }
        BufferedOutputStream os = new BufferedOutputStream(
                new FileOutputStream(f));

        int c;
        while ((c = is.read()) != -1) {
            os.write((byte) c);
        }
        is.close();
        os.close();

        return f;
    }

    /**
     * the default parseJarFile method
     */
    private void parseJarFile() {
        parseJarFile(new File(parentPath));
    }

    /**
     * Parses the jar file.
     * @param file
     *            the file to be parsed, which should be jar file, otherwise, an
     *            ioexception will be thrown.
     */
    private void parseJarFile(final File file) {
        if (file == null) {
            throw new RuntimeException("file is null.");
        }

        JarFile jarFile = null;
        try {
            jarFile = new JarFile(file);
            Enumeration entries = jarFile.entries();

            while (entries.hasMoreElements()) {
                ZipEntry entry = (ZipEntry) entries.nextElement();
                if (entry.isDirectory()) {
                    continue;
                }

                String entryName = entry.toString();
                if (entryName == null) {
                    continue;
                } else if (entryName.endsWith(".jar")) {
                    // Found a child jar file inside the parent.
                    File f = extractJarFileFromJar(jarFile, entry);
                    if (f != null) {
                        // Try to extract descendant jar files from the child
                        // jar recursively.
                        parseJarFile(f);
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (jarFile != null) {
                try {
                    jarFile.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }
}