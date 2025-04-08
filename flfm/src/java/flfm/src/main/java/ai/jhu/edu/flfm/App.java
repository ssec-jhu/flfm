package ai.jhu.edu.flfm;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Paths;

import ij.ImagePlus;
import ij.ImageStack;
import ij.io.FileSaver;
import ij.io.Opener;
import ij.process.ImageProcessor;

/**
 * Hello world!
 */
public class App {


    public static NDArray openArrayIJ(String fname, NDManager manager) throws Exception {
        Opener opener = new Opener();

        ImagePlus image = opener.openImage(fname);

        int width = image.getWidth();
        int height = image.getHeight();
        int numSlices = image.getStackSize();
        System.out.println("Loaded a multipage TIFF with dimensions: " + width + "x" + height + " and " + numSlices + " slices.");

        // Create an NDArray to hold the entire stack
        NDArray fullStack = manager.zeros(new Shape(numSlices, height, width));

        // Process each slice
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

        // Print NDArray info
        System.out.println("NDArray stack shape: " + fullStack.getShape());

        return fullStack;
    }

    public static void saveArrayIJ(String fname, NDArray array) throws Exception {

            // Convert NDArray back to a multipage TIFF
            ImageStack stack = new ImageStack(
                (int)array.getShape().get(1),
                (int)array.getShape().get(2)
            );
            int numSlices = (int)array.getShape().get(0);


            for (int slice = 0; slice < numSlices; slice++) {
                // Get each slice from the NDArray (convert to 2D)
                NDArray sliceNDArray = array.get(slice);
                float[] slicePixels = sliceNDArray.toFloatArray();

                // Convert float pixels to short (16-bit) array
                short[] shortPixels = new short[slicePixels.length];
                for (int i = 0; i < slicePixels.length; i++) {
                    shortPixels[i] = (short) slicePixels[i];
                }

                // Add this slice to the ImageStack
                stack.addSlice(null, shortPixels);
            }

            // Create an ImagePlus from the stack
            ImagePlus image = new ImagePlus("Multipage TIFF", stack);

            // Save the image as a multipage TIFF file
            FileSaver fileSaver = new FileSaver(image);
            fileSaver.saveAsTiffStack(fname);

            System.out.println("Multipage TIFF saved successfully!");
    }

    public static void main(String[] args) throws Exception, IOException, MalformedModelException, TranslateException  {
        String imagePath = "/home/ryanhausen/repos/flfm/data/yale/light_field_image.tif";
        String psfPath = "/home/ryanhausen/repos/flfm/data/yale/measured_psf.tif";
        String modelPath = "/home/ryanhausen/repos/flfm/notebooks/richardson_lucy_10.pt";

        try(NDManager manager = NDManager.newBaseManager()){
            NDArray image = openArrayIJ(imagePath, manager);
            System.out.println("Image loaded!");
            NDArray psf = openArrayIJ(psfPath, manager);
            NDArray normedPsf = psf.div(psf.sum());
            System.out.println("PSF loaded!");

            try (Model model = Model.newInstance("flfm", "PyTorch")) {
                System.out.println("Loading model...");
                model.load(Paths.get(modelPath));
                System.out.println("Model loaded!");

                try (Predictor<NDArray[], NDArray> predictor = model.newPredictor(new MyTranslator())) {
                    System.out.println("Predicting...");
                    // time the call
                    long start = System.currentTimeMillis();
                    NDArray out = predictor.predict(new NDArray[]{image, normedPsf});
                    long end = System.currentTimeMillis();
                    System.out.println("Prediction took " + (end - start) + " ms");
                    saveArrayIJ("output_file.tiff", out);
                }
            }
        }
    }
}
