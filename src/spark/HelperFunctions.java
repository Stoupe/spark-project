package spark;

import java.io.File;

public class HelperFunctions {

    /**
     * Recursively delete a directory and all files within
     *
     * thanks https://mkyong.com/java/how-to-delete-directory-in-java/
     * @param file the directory to delete
     */
    public static void deleteDirectory(File file) {

        File[] list = file.listFiles();
        if (list != null) {
            for (File temp : list) {
                //recursive delete
                System.out.println("Visit " + temp);
                deleteDirectory(temp);
            }
        }

//        if (file.delete()) {
//            System.out.printf("Delete : %s%n", file);
//        } else {
//            System.err.printf("Unable to delete file or directory : %s%n", file);
//        }

    }

}
