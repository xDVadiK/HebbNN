using System;
using System.Collections.Generic;
using System.Drawing;
using System.Reflection;
using System.Windows.Forms;

namespace HebbNN
{
    public partial class Form1 : Form
    {
        List<Button> referenceImages;
        List<Button> recognizedImage;
        List<Button> providedImage;

        public Form1()
        {
            InitializeComponent();
            referenceImages = new List<Button>();
            recognizedImage = new List<Button>();
            providedImage = new List<Button>();
            SavingImageElements();
            comboBox1.SelectedIndex = 0;
        }

        private void SavingImageElements()
        {
            for (int i = 1; i <= 25 * 6; i++)
            {
                String elementName = "button" + i;
                Type type = typeof(Form1);
                FieldInfo field = type.GetField(elementName, BindingFlags.NonPublic | BindingFlags.Instance);
                Button element = (Button)field.GetValue(this);
                if (i < 26)
                {
                    providedImage.Add(element);
                    element.Click += new EventHandler(ChangeColor);
                }
                else if (i < 51)
                {
                    recognizedImage.Add(element);
                }
                else
                {
                    referenceImages.Add(element);
                }
            }
        }

        public void ChangeColor(object sender, EventArgs e)
        {
            Button button = sender as Button;
            if (button.BackColor == Color.Black)
            {
                button.BackColor = SystemColors.ControlLight;
            }
            else
            {
                button.BackColor = Color.Black;
            }
        }

        private void StartButton_Click(object sender, EventArgs e)
        {
            try
            {
                Hebb hebbNN = new Hebb(referenceImages, recognizedImage, recognizedImage.Count);
                if (!hebbNN.Recognition(providedImage))
                {
                    MessageBox.Show("Сеть Хебба не смогла определить к какому из эталонных изображений наиболее близко предъявленное изображение", "Информация");
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "Ошибка");
            }
        }

        private void comboBox1_SelectedIndexChanged(object sender, EventArgs e)
        {
            switch (comboBox1.SelectedIndex)
            {
                case 0:
                    int[] array1 = { -1, 1, 1, 1, -1,
                        1, -1, -1, -1, 1,
                        -1, -1, -1, -1, -1,
                        -1, -1, -1, -1, -1,
                        -1, -1, -1, -1, -1,
                    };
                    RenderingProvidedImage(array1);
                    break;
                case 1:
                    int[] array2 = { 1, -1, -1, -1, 1,
                        1, -1, -1, -1, 1,
                        1, -1, -1, -1, 1,
                        1, -1, -1, -1, 1,
                        1, -1, -1, -1, 1
                    };
                    RenderingProvidedImage(array2);
                    break;
                case 2:
                    int[] array3 = { -1, -1, -1, -1, -1,
                        -1, -1, -1, -1, -1,
                        -1, 1, 1, 1, -1,
                        -1, -1, -1, -1, -1,
                        1, 1, 1, -1, -1
                    };
                    RenderingProvidedImage(array3);
                    break;
                case 3:
                    foreach (Button element in providedImage)
                    {
                        element.BackColor = SystemColors.ControlLight;
                        element.Enabled = true;
                    }
                    break;
            }


        }

        private void RenderingProvidedImage(int[] array)
        {
            for (int i = 0; i < array.Length; i++)
            {
                providedImage[i].Enabled = false;
                if (array[i] == 1)
                {
                    providedImage[i].BackColor = Color.Black;
                }
                else
                {
                    providedImage[i].BackColor = SystemColors.ControlLight;
                    providedImage[i].UseVisualStyleBackColor = true;
                }
            }
        }
    }
}