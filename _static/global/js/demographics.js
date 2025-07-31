const json = {
"title": " ",
   "showCompletedPage": false,
   "showPrevButton": false,
 "completeText": "Next",
 "description": "We ask you to provide us with the following information, which will be treated confidentially and stored anonymously.",
 "pages": [
  {
   "name": "page1",
   "elements": [
    {
     "type": "radiogroup",
     "name": "gender",
     "title": "Gender:",
     "isRequired": true,
     "choices": [
      "female",
      "male",
      "diverse",
      "I prefer not to say"
     ]
    },
    {
     "type": "text",
     "name": "age",
     "title": "Age (years):",
     "isRequired": true,
     "inputType": "number"
    },
    {
     "type": "radiogroup",
     "name": "handedness",
     "title": "Handedness:",
     "isRequired": true,
     "choices": [
      "right-handed",
      "left-handed",
      "ambidextrous/two-handed"
     ]
    },
    {
      "type": "country",
     "name": "grewUpInCountry",
     "title": "Grew up in country:",
     "isRequired": true
    },
    {
     "type": "country",
     "name": "currentlyLivingInCountry",
     "title": "Currently living in country:",
     "isRequired": true
    },
    {
     "type": "radiogroup",
     "name": "nativeLanguage",
     "title": "Native language:",
     "isRequired": true,
     "choices": [
      "English",
      "other"
     ]
    },
    {
     "type": "dropdown",
     "name": "education",
     "title": "Completed highest level of education:",
     "isRequired": true,
     "choices": [
      "Less than a high school diploma",
      "High school degree or equivalent (e.g. GED)",
      "Some college, no degree",
      "Associate degree (e.g. AA, AS)",
      "Bachelor's degree (e.g. BA, BS)",
      "Master's degree (e.g. MA, MS, MEd)",
      "Doctorate or professional degree (e.g. MD, DDS, PhD)"
     ]
    }
   ]
  }
 ]
}
//Register new "country" component
Survey.ComponentCollection.Instance.add({
    //Unique component name. It becomes a new question type. Please note, it should be written in lowercase.
    name: "country",
    //The text that shows on toolbox
    title: "Country",
    //The actual question that will do the job
    questionJSON: {
        type: "dropdown",
        placeholder: "Select a country...",
        choicesByUrl: {
            url: "https://surveyjs.io/api/CountriesExample",
        },
    },
});
const survey = new Survey.Model(json);

survey.onComplete.add((sender, options) => {
    var surveyData = JSON.stringify(sender.data, null, 3);
    $('#survey_data').val(surveyData);

    $('#form').submit();
});

$("#surveyElement").Survey({model: survey});