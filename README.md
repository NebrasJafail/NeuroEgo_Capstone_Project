## NeuroEgo_Capstone_Project
# NeuroEgo: EEG Emotion Reader (Prototype)

A first prototype for NeuroEgo, an idea to help people with disabilities communicate through brain signals.

The app takes an EEG sample, predicts the emotion (positive, neutral, or negative) with a Random Forest model, and reads a matching sentence aloud.

**Note:** this is a simulated demo. It uses recorded samples from a public Kaggle dataset (`emotions.csv`), not a real headband.

**Result:** 98.8% test accuracy (random guessing would be about 33%). This is preliminary.

## Run it

```
pip install -r requirements.txt
streamlit run emotions_app.py
```
The Website Link: https://neuroegocapstoneproject-mbkfi4posx75bcz526ezzc.streamlit.app/

Keep `emotions.csv` in the same folder as the app.

## Next steps

- Test with real EEG hardware
- Move from emotions to commands and phrases

Built by Nebras Jafail for the General Assembly Data Science capstone.
