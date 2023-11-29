using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;

namespace HebbNN
{
    internal class Hebb
    {
        private int imageSize;
        private List<double[]> weights;
        private List<Button> recognizedImage;
        List<Tuple<double[], double[]>> trainingData;
        
        public Hebb(List<Button> referenceImages, List<Button> recognizedImage, int imageSize)
        {
            trainingData = new List<Tuple<double[], double[]>>();

            this.recognizedImage = recognizedImage;

            int shift = 0;
            for (int i = 0; i < referenceImages.Count / imageSize; i++)
            {
                double[] t = new double[] { -1, -1, -1, -1 };
                t[i] += 2;
                double[] data = new double[imageSize];
                for (int j = 0; j < imageSize; j++)
                {
                    Button button = referenceImages[j + shift];
                    data[j] = button.BackColor == Color.Black ? 1 : -1;
                }
                trainingData.Add(new Tuple<double[], double[]>(data, t));
                shift += imageSize;
            }

            this.weights = new List<double[]>();
            this.imageSize = imageSize;

            Train(trainingData);
        }

        public bool Recognition(List<Button> providedImage)
        {
            double[] image = new double[providedImage.Count];
            for(int i = 0; i < image.Length; i++)
            {
                image[i] = providedImage[i].BackColor == Color.Black ? 1 : -1;
            }
            var response = GetResponse(image);
            if(response.Count(x => x == 1) == 1)
            {
                for (var i = 0; i < response.Length; i++)
                {
                    if (response[i] == 1)
                    {
                        for (int j = 0; j < imageSize; j++)
                        {
                            if (trainingData[i].Item1[j] == 1)
                            {
                                recognizedImage[j].BackColor = Color.Black;
                            }
                            else
                            {
                                recognizedImage[j].UseVisualStyleBackColor = true;
                            }
                        }
                        return true;
                    }
                }
            }
            return false;
        }

        private double[] GetResponse(double[] image)
        {
            var response = new double[weights.Count];
            for (var imageIndex = 0; imageIndex < weights.Count; imageIndex++)
            {
                response[imageIndex] = weights[imageIndex][0];
                for (var neuron = 1; neuron <= imageSize; neuron++)
                {
                    response[imageIndex] += image[neuron - 1] * weights[imageIndex][neuron];
                }
            }
            return response.Select(r => r > 0 ? 1.0 : -1.0).ToArray();
        }

        public void Train(List<Tuple<double[], double[]>> trainingData)
        {
            for (var i = 0; i < trainingData.Count; i++)
            {
                weights.Add(Enumerable.Repeat(0.0, imageSize + 1).ToArray());
            }

            for (var i = 0; i < trainingData.Count; i++)
            {
                while (!CheckResponses(GetResponse(trainingData[i].Item1), trainingData[i].Item2))
                {
                    for (var imageIndex = 0; imageIndex < weights.Count; imageIndex++)
                    {
                        weights[imageIndex][0] += trainingData[i].Item2[imageIndex];
                        for (var neuron = 1; neuron <= imageSize; neuron++)
                        {
                            weights[imageIndex][neuron] += trainingData[i].Item1[neuron - 1] * trainingData[i].Item2[imageIndex];
                        }
                    }
                }
            }
        }

        private bool CheckResponses(double[] a, double[] b)
        {
            if (a.Length != b.Length) return false;

            for (var i = 0; i < a.Length; i++)
            {
                if (a[i] != b[i]) return false;
            }
            return true;
        }
    }
}