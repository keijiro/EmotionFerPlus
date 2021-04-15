using UnityEngine;
using Unity.Barracuda;
using System.Linq;

namespace EmotionFerPlus {

sealed class Test : MonoBehaviour
{
    #region Editable attributes

    [SerializeField] Texture2D _image = null;
    [SerializeField] NNModel _model = null;
    [SerializeField] ComputeShader _preprocessor = null;
    [SerializeField] UnityEngine.UI.RawImage _preview = null;
    [SerializeField] UnityEngine.UI.Text _label = null;

    #endregion

    #region Compile-time constants

    const int ImageSize = 64;

    readonly static string[] Labels =
      { "Neutral", "Happiness", "Surprise", "Sadness",
        "Anger", "Disgust", "Fear", "Contempt"};

    #endregion

    #region MonoBehaviour implementation

    void Start()
    {
        using var worker = ModelLoader.Load(_model).CreateWorker();

        // Preprocessing
        using var preprocessed = new ComputeBuffer(ImageSize * ImageSize, sizeof(float));
        _preprocessor.SetTexture(0, "_Texture", _image);
        _preprocessor.SetBuffer(0, "_Tensor", preprocessed);
        _preprocessor.Dispatch(0, ImageSize / 8, ImageSize / 8, 1);

        // Emotion recognition model
        using (var tensor = new Tensor(1, ImageSize, ImageSize, 1, preprocessed))
            worker.Execute(tensor);

        // Output aggregation
        var probs = worker.PeekOutput().AsFloats().Select(x => Mathf.Exp(x));
        var sum = probs.Sum();
        var lines = Labels.Zip(probs, (l, p) => $"{l,-12}: {p / sum:0.00}");
        _label.text = string.Join("\n", lines);

        _preview.texture = _image;
    }

    #endregion
}

} // namespace EmotionFerPlus
