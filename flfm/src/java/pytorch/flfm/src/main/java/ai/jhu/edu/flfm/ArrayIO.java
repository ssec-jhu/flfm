package ai.jhu.edu.flfm;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.index.NDIndex;

import ij.ImagePlus;
import ij.ImageStack;
import ij.io.FileSaver;
import ij.io.Opener;
import ij.process.ImageProcessor;


public class ArrayIO {

    /**
     * Load a TIFF image stack into an NDArray.
     *
     * @param fname   The file name of the TIFF image stack.
     * @param manager The NDManager to manage the NDArray.
     * @return An NDArray representing the image stack.
     */
    public static NDArray loadArray(String fname, NDManager manager) {
        ImagePlus image = new Opener().openImage(fname);

        int width = image.getWidth();
        int height = image.getHeight();
        int numSlices = image.getStackSize();

        // Create an NDArray to hold the entire stack
        NDArray fullStack = manager.zeros(new Shape(numSlices, height, width));
        // slices are one indexed in ImageJ, but zero indexed in NDArray
        for (int slice = 1; slice <= numSlices; slice++) {
            ImageProcessor processor = image.getStack().getProcessor(slice);
            short[] pixels = (short[]) processor.getPixels();

            // Convert slice to a float array
            float[] floatPixels = new float[pixels.length];
            for (int i = 0; i < pixels.length; i++) {
                floatPixels[i] = pixels[i] & 0xFFFF; // Unsigned conversion
            }

            // Add slice to the NDArray stack
            NDArray sliceNDArray = manager.create(floatPixels, new Shape(height, width));
            fullStack.set(new NDIndex(slice-1+",:,:"), sliceNDArray); // Set the slice into the stack (0-indexed)
        }

        return fullStack;
    }

    /**
     * Save an NDArray as a TIFF image stack.
     *
     * @param fname  The file name to save the TIFF image stack.
     * @param array  The NDArray to save.
     */
    public static void saveArray(String fname, NDArray array) {

        ImageStack stack = new ImageStack(
            (int)array.getShape().get(1),
            (int)array.getShape().get(2)
        );
        int numSlices = (int)array.getShape().get(0);

        for (int slice = 0; slice < numSlices; slice++) {
            // Get each slice from the NDArray (convert to 2D)
            NDArray sliceNDArray = array.get(slice);
            float[] slicePixels = sliceNDArray.toFloatArray();

            short[] pixels = new short[slicePixels.length];
            for (int i = 0; i < slicePixels.length; i++) {
                pixels[i] = (short) slicePixels[i];
            }

            stack.addSlice(null, pixels);
        }

        // Create a new ImagePlus object with the stack
        ImagePlus image = new ImagePlus("Image", stack);

        // Save the image as a TIFF file
        FileSaver fileSaver = new FileSaver(image);
        fileSaver.saveAsTiffStack(fname);
    }
}
