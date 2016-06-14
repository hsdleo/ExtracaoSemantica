package app;

import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
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

import net.sourceforge.tess4j.ITessAPI.TessBaseAPI;
import net.sourceforge.tess4j.ITesseract;
import net.sourceforge.tess4j.Tesseract1;
import net.sourceforge.tess4j.TesseractException;

public class CaptureData {
	public TransparentFrame windowRefence;
	private static String ocr = "";
	private static String pathFile = "C:\\semantica\\";
	// private static String pathFile = "";
	private static String nameFile = "saida.png";
	private static List<String> lista_leitura = new ArrayList<>();

	public void TakePicture() {
		try {
			System.out.println("Entrei no take");

			// getting width and height of image
			for (int index = 1; index <= 1959; index++) {
				// "%08d%n"
				System.out.println(index);
				String indice = String.format("%04d", index);
				String caminho = pathFile + "frames_" + indice + ".png";
				crop(caminho);
				BufferedImage img = ImageIO.read(new File(pathFile + nameFile));
				BufferedImage bimg = null;
				try {

					bimg = Resize(img, (int) (img.getWidth() * 1), (int) (img.getHeight() * 1));
					// ImageIO.write(bimg, "png", new File(pathFile +
					// nameFile));
					ImageIO.write(bimg, "PNG", new File(pathFile + nameFile));
					int a = ProcessOCR();

				} catch (Exception e) {
					e.printStackTrace();
				}
			}
			// SALVAR NO TXT

			File saida_texto = new File("C:\\semantica\\final.txt");
			try {
				saida_texto.delete();
				saida_texto.createNewFile();

				BufferedWriter out = Files.newBufferedWriter(Paths.get("C:\\semantica\\final.txt"),
						StandardCharsets.UTF_8, StandardOpenOption.APPEND);
				System.out.println("Tamanho = "+ lista_leitura.size());
				for (String ocr_ : lista_leitura) {
					out.append(ocr_);
					out.newLine();
				}
				out.close();

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
						// System.out.println("Consegui");
					} catch (TesseractException e) {
						lista_leitura.add("");

						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
			}, 300, TimeUnit.MILLISECONDS);
		} catch (TimeoutException e) {
			// System.out.println("Nao deu2");
		} finally {
			if(ocr.length()>45)lista_leitura.add("");
			else lista_leitura.add(ocr);
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
				System.out.println(
						"(" + x + "," + y + ") = " + pixel.getRed() + "," + pixel.getGreen() + "," + pixel.getBlue());

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
				/*
				 * if (pixel.getRed() >= 170 && pixel.getGreen() >= 170 &&
				 * pixel.getBlue() >= 170) bmap.setRGB(x, y,
				 * Color.WHITE.getRGB());
				 */
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

		int x = 95;
		int y = 35;
		int w = 525;
		int h = 85;

		try {
			BufferedImage image = ImageIO.read(new File(caminho));
			AffineTransform at = new AffineTransform();
            at.rotate(0.05,image.getWidth()/2,image.getHeight()/2);
            AffineTransformOp op = new AffineTransformOp(at,
                    AffineTransformOp.TYPE_BILINEAR);
            image = op.filter(image, null);
			BufferedImage out = image.getSubimage(x, y, w, h);
			
            /*Graphics2D g2d =  image.createGraphics();
            g2d.drawImage(image,at,null);*/
			ImageIO.write(out, "jpg", new File(pathFile + nameFile));

		} catch (IOException e) {
			System.out.println("Erro leitura imagem");
			System.out.println(e.getMessage());

		}

	}

}
