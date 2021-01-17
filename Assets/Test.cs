using UnityEngine;
using Unity.Barracuda;
using System.Linq;

namespace EmotionFerPlus {

sealed class Test : MonoBehaviour
{
    #region Editable attributes

    [SerializeField] NNModel _model = null;
    [SerializeField] UnityEngine.UI.RawImage _preview = null;
    [SerializeField] UnityEngine.UI.Text _label = null;
    [SerializeField, HideInInspector] ComputeShader _preprocessor = null;

    #endregion

    #region Internal objects

    WebCamTexture _webcamRaw;
    RenderTexture _webcamBuffer;
    ComputeBuffer _preprocessed;
    IWorker _worker;

    const int ImageSize = 64;

    readonly static string[] Labels =
      { "Neutral", "Happiness", "Surprise", "Sadness",
        "Anger", "Disgust", "Fear", "Contempt"};

    #endregion

    #region MonoBehaviour implementation

    void Start()
    {
        _webcamRaw = new WebCamTexture();
        _webcamBuffer = new RenderTexture(512, 512, 0);
        _preprocessed = new ComputeBuffer(ImageSize * ImageSize, sizeof(float));
        _worker = ModelLoader.Load(_model).CreateWorker();

        _webcamRaw.Play();
        _preview.texture = _webcamBuffer;
    }

    void OnDisable()
    {
        _preprocessed?.Dispose();
        _preprocessed = null;

        _worker?.Dispose();
        _worker = null;
    }

    void OnDestroy()
    {
        if (_webcamRaw != null) Destroy(_webcamRaw);
        if (_webcamBuffer != null) Destroy(_webcamBuffer);
    }

    void Update()
    {
        // Cropping
        var scale = new Vector2((float)_webcamRaw.height / _webcamRaw.width, 1);
        var offset = new Vector2(scale.x / 2, 0);
        Graphics.Blit(_webcamRaw, _webcamBuffer, scale, offset);

        // Preprocessing
        _preprocessor.SetTexture(0, "_Texture", _webcamBuffer);
        _preprocessor.SetBuffer(0, "_Tensor", _preprocessed);
        _preprocessor.Dispatch(0, ImageSize / 8, ImageSize / 8, 1);

        // Emotion recognition model
        using (var tensor = new Tensor(1, ImageSize, ImageSize, 1, _preprocessed))
            _worker.Execute(tensor);

        // Output aggregation
        var probs = _worker.PeekOutput().AsFloats().Select(x => Mathf.Exp(x));
        var sum = probs.Sum();
        var lines = Labels.Zip(probs, (l, p) => $"{l,-12}: {p / sum:0.00}");
        _label.text = string.Join("\n", lines);
    }

    #endregion
}

} // namespace EmotionFerPlus
