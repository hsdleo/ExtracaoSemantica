package app;

import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import javax.imageio.ImageIO;
import javax.swing.JOptionPane;

import org.apache.commons.io.FileUtils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javafx.scene.control.CheckBox;
import javafx.scene.control.Slider;
import net.sourceforge.tess4j.ITessAPI.TessBaseAPI;
import net.sourceforge.tess4j.ITesseract;
import net.sourceforge.tess4j.Tesseract1;
import net.sourceforge.tess4j.TesseractException;

public class CaptureData {
	private CheckBox canny;
	// canny threshold value
	private Slider threshold;
	private CheckBox dilateErode;
	private CheckBox inverse;
	private List<Mat> planes = new ArrayList<>();
	private static int index_atual = 0;
	public TransparentFrame windowRefence;
	private static String ocr = "";
	private static String pathFile = "C:\\semantica\\";
	// private static String pathFile = "";
	private static String nameFile = "saida.png";
	private static List<String> lista_leitura = new ArrayList<>();
	private static List<Integer> lista_index = new ArrayList<>();

	private static List<String> lista_pixels = new ArrayList<>();

	public void TakePicture() {
		try {
		
			System.out.println("Entrei no take");

			// getting width and height of image
			for (int index = 1; index <= 21; index++) {
				index_atual = index;
				// "%08d%n"
				System.out.println(index);
				String indice = String.format("%04d", index);
				String caminho = pathFile + "frames_" + indice + ".png";
				//String caminho = pathFile + "extra.png";
				crop(caminho);
				// apply Otsu threshold
				/*
				 * Mat bw = new Mat(im.size(), CvType.CV_8U);
				 * Imgproc.threshold(im, bw, 0, 255, Imgproc.THRESH_BINARY_INV |
				 * Imgproc.THRESH_OTSU);
				 */
				/*
				 * Mat dist = new Mat(im.size(), CvType.CV_32F);
				 * Imgproc.distanceTransform(bw, dist, Imgproc.CV_DIST_L2,
				 * Imgproc.CV_DIST_MASK_PRECISE); // threshold the distance
				 * transform Mat dibw32f = new Mat(im.size(), CvType.CV_32F);
				 * final double SWTHRESH = 8.0; // stroke width threshold
				 * Imgproc.threshold(dist, dibw32f, SWTHRESH/2.0, 255,
				 * Imgproc.THRESH_BINARY); Mat dibw8u = new Mat(im.size(),
				 * CvType.CV_8U); dibw32f.convertTo(dibw8u, CvType.CV_8U);
				 * 
				 * Mat kernel =
				 * Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3,
				 * 3)); // open to remove connections to stray elements Mat cont
				 * = new Mat(im.size(), CvType.CV_8U);
				 * Imgproc.morphologyEx(dibw8u, cont, Imgproc.MORPH_OPEN,
				 * kernel); // find contours and filter based on bounding-box
				 * height final double HTHRESH = im.rows() * 0.5; //
				 * bounding-box height threshold List<MatOfPoint> contours = new
				 * ArrayList<MatOfPoint>(); List<Point> digits = new
				 * ArrayList<Point>(); // contours of the possible digits
				 * Imgproc.findContours(cont, contours, new Mat(),
				 * Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE); for (int i
				 * = 0; i < contours.size(); i++) { if
				 * (Imgproc.boundingRect(contours.get(i)).height > HTHRESH) { //
				 * this contour passed the bounding-box height threshold. add it
				 * to digits digits.addAll(contours.get(i).toList()); } } //
				 * find the convexhull of the digit contours MatOfInt
				 * digitsHullIdx = new MatOfInt(); MatOfPoint hullPoints = new
				 * MatOfPoint(); hullPoints.fromList(digits);
				 * Imgproc.convexHull(hullPoints, digitsHullIdx); // convert
				 * hull index to hull points List<Point> digitsHullPointsList =
				 * new ArrayList<Point>(); List<Point> points =
				 * hullPoints.toList(); for (Integer i: digitsHullIdx.toList())
				 * { digitsHullPointsList.add(points.get(i)); } MatOfPoint
				 * digitsHullPoints = new MatOfPoint();
				 * digitsHullPoints.fromList(digitsHullPointsList); // create
				 * the mask for digits List<MatOfPoint> digitRegions = new
				 * ArrayList<MatOfPoint>(); digitRegions.add(digitsHullPoints);
				 * Mat digitsMask = Mat.zeros(im.size(), CvType.CV_8U);
				 * Imgproc.drawContours(digitsMask, digitRegions, 0, new
				 * Scalar(255, 255, 255), -1); // dilate the mask to capture any
				 * info we lost in earlier opening
				 * Imgproc.morphologyEx(digitsMask, digitsMask,
				 * Imgproc.MORPH_DILATE, kernel); // cleaned image ready for OCR
				 * Mat cleaned = Mat.zeros(im.size(), CvType.CV_8U);
				 * dibw8u.copyTo(cleaned, digitsMask);
				 */
				// feed cleaned to Tesseract
				BufferedImage img = ImageIO.read(new File(pathFile + nameFile));
				BufferedImage bimg = null;
				try {
					bimg = Resize(img, (int) (img.getWidth() * 2), (int) (img.getHeight() * 2));
					ImageIO.write(bimg, "PNG", new File(pathFile + nameFile));

				} catch (Exception e) {
				}
				//Mat image = Imgcodecs.imread(pathFile + nameFile,Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
				 Mat image = Imgcodecs.imread(pathFile + nameFile);

				//Imgproc.cvtColor(image, image, Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
				Imgproc.blur(image, image, new Size(3, 3));
				//Imgproc.threshold(image, image, 25, 255, Imgproc.THRESH_BINARY);
				//Imgproc.threshold(image, image, 30, 255, Imgproc.THRESH_BINARY);
				//int a = ProcessOCR();
				
				//image = doCanny(image);

				//Imgcodecs.imwrite(pathFile +nameFile, image);
				Mat gray = new Mat();
				//Imgproc.cvtColor(image, gray, Imgproc.COLOR_RGBA2GRAY);
				//Imgproc.Canny(gray, gray, 50, 200);
				//Imgproc.threshold(gray, gray, 0, 255, Imgproc.THRESH_BINARY);
				//Imgproc.Canny(gray, gray, 50, 200);
				//Imgcodecs.imwrite(pathFile +nameFile, gray);
				/*
				List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
				Mat hierarchy = new Mat();
				Imgproc.findContours(gray, contours, hierarchy, Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);
				System.out.println(contours.size());
				Mat a = Imgcodecs.imread(pathFile + "branco.png");
				for (int contourIdx = 0; contourIdx < contours.size(); contourIdx++) {
				    Imgproc.drawContours(gray, contours, contourIdx, new Scalar(255, 255, 255),1);
				}
				*/

				//Imgcodecs.imwrite(pathFile +nameFile, gray);
				Imgcodecs.imwrite(pathFile +nameFile, image);

				//Imgcodecs.imwrite(pathFile +"saida2.png", image);
				int aa = ProcessOCR();

				
				/*	
				Mat padded = optimizeImageDim(image);

				padded.convertTo(padded, CvType.CV_32F);

				// prepare the image planes to obtain the complex image
				planes.add(padded);
				System.out.println("imprimindo1");

				planes.add(Mat.zeros(padded.size(), CvType.CV_32F));
				System.out.println("imprimindo2");


				// prepare a complex image for performing the dft
				Mat complex = new Mat();
				Core.merge(planes, complex);
				System.out.println("imprimindo3");

				// dft
				Core.dft(complex, complex);
				Mat complex2 = new Mat();

				System.out.println("imprimindo4");


				// optimize the image resulting from the dft operation
				Mat magnitude = createOptimizedMagnitude(complex);				
				Imgcodecs.imwrite(pathFile +"saida1.png", magnitude);
*/				
				//Imgproc.blur(image, image, new Size(3, 3));
				
				//Imgproc.threshold(image, image, 30, 255, Imgproc.THRESH_BINARY_INV);
				//Imgcodecs.imwrite(pathFile +nameFile, image);
/*
				Mat lines = new Mat();
				//Imgproc.HoughLinesP(image, lines, rho, theta, threshold, minLineLength, maxLineGap);()
				Imgproc.HoughLinesP(magnitude, lines, 1, Math.PI /180, 0, 5, 1);
				for(int i = 0; i < lines.cols(); i++) {
					double[] val = lines.get(0, i);
				
					Imgproc.line(img2, new Point(val[0], val[1]), new Point(val[2], val[3]), new Scalar(0, 0, 255), 2);
				}
				System.out.println("imprimindo5");

				Imgcodecs.imwrite(pathFile + "saidaFinal.png", img2);
				System.out.println("imprimindo6");

				boolean primeira = true;
				int p1_linha = 0;
				int p1_coluna = 0;
				int p2_linha = 0;
				int p2_coluna = 0;
				for (int r = 0; r < magnitude.rows(); r++) {
					for (int c = 0; c < magnitude.cols(); c++) {
						if (magnitude.get(r, c)[0] == 255) {
							if (primeira) {
								primeira = false;
								p1_linha = r;
								p1_coluna = c;
							} else {
								p2_linha = r;
								p2_coluna = c;
							}

						}
					}
				}
				*/
				/*System.out.println("imprimindo");
				System.out.println(p1_linha + ","+p1_coluna + "," + p2_linha + ","+p2_coluna);
				System.out.println(Math.toDegrees(Math.atan(teste)));

				System.out.println(Math.toDegrees(Math.atan((p1_coluna - p2_coluna) / (p2_linha - p1_linha))));
				 */
				//Imgcodecs.imwrite(pathFile + nameFile, magnitude);
				//Imgcodecs.imwrite(pathFile + "saidaFinal", img2);

				/*
				 * Mat img_teste = Imgcodecs.imread("resources/teste.png");
				 * MatOfByte buffer = new MatOfByte();
				 * Imgproc.cvtColor(img_teste, img_teste,
				 * Imgproc.COLOR_BGR2GRAY);
				 * 
				 * //Imgcodecs.imencode(".png", img_teste, buffer); img_teste =
				 * cleanImage(img_teste);
				 * Imgcodecs.imwrite("resources/imagem_teste.png", img_teste);
				 */

			}
			// SALVAR NO TXT

			File saida_texto = new File("C:\\semantica\\final.txt");
			//File saida_pixel = new File("C:\\semantica\\pixel.txt");

			try {
				saida_texto.delete();
				saida_texto.createNewFile();
				//saida_pixel.delete();
				//saida_pixel.createNewFile();

				BufferedWriter out = Files.newBufferedWriter(Paths.get("C:\\semantica\\final.txt"),
						StandardCharsets.UTF_8, StandardOpenOption.APPEND);
				//BufferedWriter out2 = Files.newBufferedWriter(Paths.get("C:\\semantica\\pixel.txt"),
						//StandardCharsets.UTF_8, StandardOpenOption.APPEND);
				System.out.println("Tamanho = " + lista_leitura.size());
				int i = 0;
				for (String ocr_ : lista_leitura) {
					String indice = String.format("%04d", lista_index.get(i));
					String escrever = ocr_ + ";frames_" + indice + ".png";
					i++;
					out.append(escrever);
					out.newLine();
				}
				out.close();
				/*for (String pixel : lista_pixels) {
					out2.append(pixel);
					out2.newLine();
				}
				out2.close();*/

			} catch (IOException e) {
				e.printStackTrace();
			}

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public int ProcessOCR() throws Exception {
		// Call the tesseract.exe OCR
		ocr = "";

		int code = 0;
		try {
			TimeLimitedCodeBlock.runWithTimeout(new Runnable() {
				@Override
				public void run() {
					try {
						// System.out.println("Tentei");
						ITesseract instance = new Tesseract1();
						ocr = instance.doOCR(new File(pathFile + nameFile));
						lista_index.add(index_atual);
						//System.out.println("Leitura = " + ocr);
						if (ocr.length() > 45)
							lista_leitura.add("");
						else
							lista_leitura.add(ocr);
						 System.out.println("Consegui");
					} catch (TesseractException e) {
						 System.out.println("Não cosnegui 1");

						lista_leitura.add("");
						lista_index.add(index_atual);
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
			}, 1500, TimeUnit.MILLISECONDS);
		} catch (TimeoutException e) {
		  System.out.println("Nao deu2");
		} 

		return 1;
	}

	private int validationData(String dataOcr) {
		dataOcr = dataOcr.trim();
		System.out.println("DataOCR = " + dataOcr);
		int code = 0;
		if (dataOcr.length() > 3) {
			try {
				code = Integer.parseInt(dataOcr);

				return code;
			} catch (Exception e) {
				return code;
			}
		} else {
			return code;
		}
	}

	// Resize
	public BufferedImage Resize(BufferedImage bmp, int newWidth, int newHeight) {
		BufferedImage temp = (BufferedImage) bmp;

		BufferedImage bmap = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_BYTE_GRAY);

		double nWidthFactor = (double) temp.getWidth() / (double) newWidth;
		double nHeightFactor = (double) temp.getHeight() / (double) newHeight;

		double fx, fy, nx, ny;
		int cx, cy, fr_x, fr_y;
		Color color1 = new Color(0, 0, 0);
		Color color2 = new Color(0, 0, 0);
		Color color3 = new Color(0, 0, 0);
		Color color4 = new Color(0, 0, 0);
		int nRed, nGreen, nBlue;

		int bp1, bp2;

		for (int x = 0; x < bmap.getWidth(); ++x) {
			for (int y = 0; y < bmap.getHeight(); ++y) {

				fr_x = (int) Math.floor(x * nWidthFactor);
				fr_y = (int) Math.floor(y * nHeightFactor);
				cx = fr_x + 1;
				if (cx >= temp.getWidth())
					cx = fr_x;
				cy = fr_y + 1;
				if (cy >= temp.getHeight())
					cy = fr_y;
				fx = x * nWidthFactor - fr_x;
				fy = y * nHeightFactor - fr_y;
				nx = 1.0 - fx;
				ny = 1.0 - fy;

				color1 = new Color(temp.getRGB(fr_x, fr_y));
				color2 = new Color(temp.getRGB(cx, fr_y));
				color3 = new Color(temp.getRGB(fr_x, cy));
				color4 = new Color(temp.getRGB(cx, cy));

				bp1 = (int) (nx * color1.getBlue() + fx * color2.getBlue());

				bp2 = (int) (nx * color3.getBlue() + fx * color4.getBlue());

				nBlue = (int) (ny * (double) (bp1) + fy * (double) (bp2));

				// Green
				bp1 = (int) (nx * color1.getGreen() + fx * color2.getGreen());

				bp2 = (int) (nx * color3.getGreen() + fx * color4.getGreen());

				nGreen = (int) (ny * (double) (bp1) + fy * (double) (bp2));

				// Red
				bp1 = (int) (nx * color1.getRed() + fx * color2.getRed());

				bp2 = (int) (nx * color3.getRed() + fx * color4.getRed());

				nRed = (int) (ny * (double) (bp1) + fy * (double) (bp2));
				// System.out.println("Valores = " + String.valueOf(nRed)+" :
				// "+String.valueOf(nGreen)+" : "+String.valueOf(nBlue));

				bmap.setRGB(x, y, new Color(nRed, nGreen, nBlue).getRGB());

			}
		}
		// bmap = SetGrayscale_LAB(bmap);
		bmap = SetGrayscale(bmap);

		// bmap = RemoveNoise(bmap);

		return bmap;
	}

	// SetGrayscale
	public BufferedImage SetGrayscale(BufferedImage img) {

		BufferedImage temp = (BufferedImage) img;
		BufferedImage bmap = clone(temp);
		Color c;
		for (int i = 0; i < bmap.getWidth(); i++) {
			for (int j = 0; j < bmap.getHeight(); j++) {
				c = new Color(bmap.getRGB(i, j));

				int gray = (int) (.299 * c.getRed() + .587 * c.getGreen() + .114 * c.getBlue());

				bmap.setRGB(i, j, new Color(gray, gray, gray).getRGB());
			}
		}
		return clone(bmap);

	}

	public BufferedImage SetGrayscale_LAB(BufferedImage img) {
		ColorSpaceConverter a = new ColorSpaceConverter();
		BufferedImage temp = (BufferedImage) img;
		BufferedImage bmap = clone(temp);
		Color c;
		for (int i = 0; i < bmap.getWidth(); i++) {
			for (int j = 0; j < bmap.getHeight(); j++) {
				c = new Color(bmap.getRGB(i, j));
				/*
				 * System.out.println( "(" + i + "," + j + ") = " + c.getRed() +
				 * "," + c.getGreen() + "," + c.getBlue());
				 */
				double[] lab2 = a.RGBtoLAB(c.getRGB());
				// double[] lab = RGBtoLAB(c.getRed(), c.getGreen(),
				// c.getBlue());
				int gray = (int) (.299 * c.getRed() + .587 * c.getGreen() + .114 * c.getBlue());
				/*
				 * System.out.println( "(" + i + "," + j + ") = " + lab2[0] +
				 * "," + lab2[1] + "," + lab2[2]);
				 */
				// double[] xyz = LABtoXYZ(lab[0], 0, 0);
				int[] rgb_novo = a.LABtoRGB(lab2[0], 0, 0);
				/*
				 * System.out.println( "(" + i + "," + j + ") = " + rgb_novo[0]
				 * + "," + rgb_novo[1] + "," + rgb_novo[2]); /* //
				 * bmap.setRGB(i, j, new Color(gray, gray, gray).getRGB());
				 */ bmap.setRGB(i, j, new Color(rgb_novo[0], rgb_novo[1], rgb_novo[2]).getRGB());

			}
		}
		return clone(bmap);

	}

	// RemoveNoise
	public BufferedImage RemoveNoise(BufferedImage img) {
		BufferedImage temp = (BufferedImage) img;
		BufferedImage bmap = clone(temp);

		for (int x = 0; x < bmap.getWidth(); x++) {
			for (int y = 0; y < bmap.getHeight(); y++) {
				Color pixel = new Color(bmap.getRGB(x, y));
				// System.out.println(
				// "(" + x + "," + y + ") = " + pixel.getRed() + "," +
				// pixel.getGreen() + "," + pixel.getBlue());
				String linha = "(" + x + "," + y + ") = " + pixel.getRed() + "," + pixel.getGreen() + ","
						+ pixel.getBlue();
				lista_pixels.add(linha);
				/*
				 * if (pixel.getRed() < 170 && pixel.getGreen() < 170 &&
				 * pixel.getBlue()< 170) bmap.setRGB(x, y,
				 * Color.BLACK.getRGB());
				 */
			}
		}

		for (int x = 0; x < bmap.getWidth(); x++) {
			for (int y = 0; y < bmap.getHeight(); y++) {
				Color pixel = new Color(bmap.getRGB(x, y));

				if (pixel.getRed() >= 100 && pixel.getGreen() >= 100 && pixel.getBlue() >= 100) {
					bmap.setRGB(x, y, Color.WHITE.getRGB());
				} else {
					bmap.setRGB(x, y, Color.BLACK.getRGB());
				}

			}
		}

		return clone(bmap);
	}

	public static final BufferedImage clone(BufferedImage image) {
		BufferedImage clone = new BufferedImage(image.getWidth(), image.getHeight(), image.getType());
		Graphics2D g2d = clone.createGraphics();
		g2d.drawImage(image, 0, 0, null);
		g2d.dispose();
		return clone;
	}

	private void crop(String caminho) {
		//wallmart
		int x = 95;
		int y = 35;
		int w = 525;
		int h = 85;
		
		/*
		int x = 20;
		int y = 160;
		int w = 590;
		int h = 60;*/
		try {
			BufferedImage image = ImageIO.read(new File(caminho));
			//System.out.println(image.getWidth() + "," + image.getHeight());

			AffineTransform at = new AffineTransform();
			at.rotate(0, image.getWidth() / 2, image.getHeight() / 2);
			AffineTransformOp op = new AffineTransformOp(at, AffineTransformOp.TYPE_BILINEAR);
			 image = op.filter(image, null);
			
			BufferedImage out = image.getSubimage(x, y, w, h);

			/*
			 * Graphics2D g2d = image.createGraphics();
			 * g2d.drawImage(image,at,null);
			 */
			ImageIO.write(out, "png", new File(pathFile + nameFile));

		} catch (IOException e) {
			System.out.println("Erro leitura imagem");
			System.out.println(e.getMessage());

		}

	}

	public Mat cleanImage(Mat srcImage) {
		Core.normalize(srcImage, srcImage, 0, 255, Core.NORM_MINMAX);
		// Imgproc.threshold(srcImage, srcImage, 0, 255, Imgproc.THRESH_OTSU);
		// Imgproc.erode(srcImage, srcImage, new Mat());
		// Imgproc.dilate(srcImage, srcImage, new Mat(), new Point(0, 0), 9);
		Point anchor = new Point(0, 0);
		// Imgproc.dilate(srcImage, srcImage, new Mat(), anchor , 9);
		return srcImage;
	}

	private Mat doBackgroundRemoval(Mat frame) {
		// init
		Mat hsvImg = new Mat();
		List<Mat> hsvPlanes = new ArrayList<>();
		Mat thresholdImg = new Mat();

		int thresh_type = Imgproc.THRESH_BINARY_INV;
		/*
		 * if (this.inverse.isSelected()) thresh_type = Imgproc.THRESH_BINARY;
		 */

		// threshold the image with the average hue value
		hsvImg.create(frame.size(), CvType.CV_8U);
		Imgproc.cvtColor(frame, hsvImg, Imgproc.COLOR_BGR2HSV);
		Core.split(hsvImg, hsvPlanes);

		// get the average hue value of the image
		double threshValue = this.getHistAverage(hsvImg, hsvPlanes.get(0));

		Imgproc.threshold(hsvPlanes.get(0), thresholdImg, threshValue, 179.0, thresh_type);

		Imgproc.blur(thresholdImg, thresholdImg, new Size(5, 5));

		// dilate to fill gaps, erode to smooth edges
		Imgproc.dilate(thresholdImg, thresholdImg, new Mat(), new Point(-1, -1), 1);
		Imgproc.erode(thresholdImg, thresholdImg, new Mat(), new Point(-1, -1), 3);

		Imgproc.threshold(thresholdImg, thresholdImg, threshValue, 179.0, Imgproc.THRESH_BINARY);

		// create the new image
		Mat foreground = new Mat(frame.size(), CvType.CV_8UC3, new Scalar(255, 255, 255));
		frame.copyTo(foreground, thresholdImg);

		return foreground;
	}

	/**
	 * Get the average hue value of the image starting from its Hue channel
	 * histogram
	 * 
	 * @param hsvImg
	 *            the current frame in HSV
	 * @param hueValues
	 *            the Hue component of the current frame
	 * @return the average Hue value
	 */
	private double getHistAverage(Mat hsvImg, Mat hueValues) {
		// init
		double average = 0.0;
		Mat hist_hue = new Mat();
		// 0-180: range of Hue values
		MatOfInt histSize = new MatOfInt(180);
		List<Mat> hue = new ArrayList<>();
		hue.add(hueValues);

		// compute the histogram
		Imgproc.calcHist(hue, new MatOfInt(0), new Mat(), hist_hue, histSize, new MatOfFloat(0, 179));

		// get the average Hue value of the image
		// (sum(bin(h)*h))/(image-height*image-width)
		// -----------------
		// equivalent to get the hue of each pixel in the image, add them, and
		// divide for the image size (height and width)
		for (int h = 0; h < 180; h++) {
			// for each bin, get its value and multiply it for the corresponding
			// hue
			average += (hist_hue.get(h, 0)[0] * h);
		}

		// return the average hue of the image
		return average = average / hsvImg.size().height / hsvImg.size().width;
	}

	/**
	 * Apply Canny
	 * 
	 * @param frame
	 *            the current frame
	 * @return an image elaborated with Canny
	 */
	private Mat doCanny(Mat frame) {
		// init
		System.out.println(frame.size());
		Mat grayImage = new Mat();
		Mat detectedEdges = new Mat();
		// grayImage.create(frame.size(), CvType.CV_8U);
		// detectedEdges.create(frame.size(), CvType.CV_8U);

		// convert to grayscale
		Imgproc.cvtColor(frame, grayImage, Imgproc.COLOR_BGR2GRAY);
		// reduce noise with a 3x3 kernel
		Imgproc.blur(grayImage, detectedEdges, new Size(3, 3));
		// this.threshold.setValue(1);
		double slider = 10;
		// canny detector, with ratio of lower:upper threshold of 3:1
		Imgproc.Canny(detectedEdges, detectedEdges, slider, slider * 3);

		// using Canny's output as a mask, display the result
		Mat dest = new Mat();
		frame.copyTo(dest, detectedEdges);
		System.out.println("porra4");

		return dest;
	}

	/**
	 * Action triggered when the Canny checkbox is selected
	 * 
	 */
	protected void cannySelected() {
		// check whether the other checkbox is selected and deselect it
		if (this.dilateErode.isSelected()) {
			this.dilateErode.setSelected(false);
			this.inverse.setDisable(true);
		}

		// enable the threshold slider
		if (this.canny.isSelected())
			this.threshold.setDisable(false);
		else
			this.threshold.setDisable(true);

		// now the capture can start
		// this.cameraButton.setDisable(false);
	}

	/**
	 * Action triggered when the "background removal" checkbox is selected
	 */

	protected void dilateErodeSelected() {
		// check whether the canny checkbox is selected, deselect it and disable
		// its slider
		if (this.canny.isSelected()) {
			this.canny.setSelected(false);
			this.threshold.setDisable(true);
		}

		if (this.dilateErode.isSelected())
			this.inverse.setDisable(false);
		else
			this.inverse.setDisable(true);

		// now the capture can start
		// this.cameraButton.setDisable(false);
	}

	public Integer getPeakElement(int[] array) {

		if (array == null || array.length == 0) {
			return null;
		}

		int n = array.length;

		int start = 0;
		int end = n - 1;

		while (start <= end) {
			int mid = (start + end) / 2;

			if ((mid == 0 || array[mid - 1] <= array[mid]) && (mid == n - 1 || array[mid] >= array[mid + 1])) {
				return array[mid]; // array[mid] is peak element
			} else if (mid > 0 && array[mid - 1] > array[mid]) {
				end = mid - 1;
			} else {
				start = mid + 1;
			}
		}
		return null;
	}

	private Mat optimizeImageDim(Mat image) {
		// init
		Mat padded = new Mat();
		// get the optimal rows size for dft
		int addPixelRows = Core.getOptimalDFTSize(image.rows());
		// get the optimal cols size for dft
		int addPixelCols = Core.getOptimalDFTSize(image.cols());
		// apply the optimal cols and rows size to the image
		
		Core.copyMakeBorder(image, padded, 0, addPixelRows - image.rows(), 0, addPixelCols - image.cols(),
				Core.BORDER_CONSTANT, Scalar.all(100));
		Imgcodecs.imwrite(pathFile +"saida7.png", padded);


		return padded;
	}

	private Mat createOptimizedMagnitude(Mat complexImage) {
		// init
		List<Mat> newPlanes = new ArrayList<>();
		Mat mag = new Mat();
		// split the comples image in two planes
		Core.split(complexImage, newPlanes);
		// compute the magnitude
		Core.magnitude(newPlanes.get(0), newPlanes.get(1), mag);

		// move to a logarithmic scale
		Core.add(Mat.ones(mag.size(), CvType.CV_32F), mag, mag);
		Core.log(mag, mag);
		// optionally reorder the 4 quadrants of the magnitude image
		shiftDFT(mag);
		// normalize the magnitude image for the visualization since both JavaFX
		// and OpenCV need images with value between 0 and 255
		// convert back to CV_8UC1
		mag.convertTo(mag, CvType.CV_8UC1);
		Core.normalize(mag, mag, 0, 255, Core.NORM_MINMAX, CvType.CV_8UC1);
		Imgcodecs.imwrite(pathFile +"saida6.png", mag);

		// Imgproc.threshold(mag, mag, 170,255, Imgproc.THRESH_BINARY_INV);
		// Imgproc.threshold(mag, mag, 60, 255, Imgproc.THRESH_BINARY);

		// you can also write on disk the resulting image...
		return mag;
	}

	private void shiftDFT(Mat image) {
		image = image.submat(new Rect(0, 0, image.cols() & -2, image.rows() & -2));
		int cx = image.cols() / 2;
		int cy = image.rows() / 2;

		Mat q0 = new Mat(image, new Rect(0, 0, cx, cy));
		Mat q1 = new Mat(image, new Rect(cx, 0, cx, cy));
		Mat q2 = new Mat(image, new Rect(0, cy, cx, cy));
		Mat q3 = new Mat(image, new Rect(cx, cy, cx, cy));

		Mat tmp = new Mat();
		q0.copyTo(tmp);
		q3.copyTo(q0);
		tmp.copyTo(q3);

		q1.copyTo(tmp);
		q2.copyTo(q1);
		tmp.copyTo(q2);
	}

}
