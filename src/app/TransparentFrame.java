package app;

import static java.awt.GraphicsDevice.WindowTranslucency.TRANSLUCENT;

import java.awt.Event;
import java.awt.GraphicsDevice;
import java.awt.GraphicsEnvironment;
import java.awt.GridBagLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;

import javax.swing.JFrame;

public class TransparentFrame extends JFrame implements MouseListener, MouseMotionListener, ActionListener {

	private static int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
	private static boolean primeiro = false;
	private static boolean segundo = false;
	private static final long serialVersionUID = 1L;

	public TransparentFrame() {
		/* Check for the transparency feature */
		GraphicsEnvironment ge = GraphicsEnvironment.getLocalGraphicsEnvironment();
		GraphicsDevice gd = ge.getDefaultScreenDevice();

		if (!gd.isWindowTranslucencySupported(TRANSLUCENT)) {
			System.err.println("Translucency is not supported");
			System.exit(0);
		}

		addMouseMotionListener(this);
		addMouseListener(this);

		setUndecorated(true);
		setLayout(new GridBagLayout());

		setSize(700, 300);
		// setSize(110, 45);

		setLocation(300, 200);
		// Set the window to 55% opaque (45% translucent).
		this.setOpacity(0.55f);

		// Display the window.
		this.setVisible(true);	
	}

	@Override
	public void actionPerformed(ActionEvent arg0) {
		// TODO Auto-generated method stub

	}

	@Override
	public void mouseDragged(java.awt.event.MouseEvent e) {
		// Positioning with the mouse
		this.setLocation(e.getLocationOnScreen().x - this.getSize().width / 2,
				e.getLocationOnScreen().y - this.getSize().height / 2);
	}

	@Override
	public void mouseMoved(MouseEvent arg0) {
		// TODO Auto-generated method stub

	}

	@Override
	public void mouseClicked(MouseEvent e) {
		if (!primeiro) {
			x1 = e.getLocationOnScreen().x;
			y1 = e.getLocationOnScreen().y;
			primeiro = true;
		} else {
			x2 = e.getLocationOnScreen().x;
			y2 = e.getLocationOnScreen().y;
			segundo = true;
			this.setSize(x2 - x1, y2 - y1);
			this.setLocation(x1, y1);

		}
		System.out.println("Clicou");
			// TODO Auto-generated method stub

	}

	@Override
	public void mouseExited(MouseEvent arg0) {
		// TODO Auto-generated method stub

	}

	@Override
	public void mousePressed(MouseEvent arg0) {
		// TODO Auto-generated method stub

	}

	@Override
	public void mouseReleased(MouseEvent arg0) {
		// TODO Auto-generated method stub

	}

	@Override
	public void mouseEntered(MouseEvent e) {
		// TODO Auto-generated method stub

	}

	public boolean retornaBool() {
		if (primeiro && segundo) {
			return true;
		} else {
			return false;
		}
	}

}
