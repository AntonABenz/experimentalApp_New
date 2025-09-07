// Demographics questionnaire structure
// Based on original benzproj documentation

window.demographicsQuestions = [
    {
        name: 'age',
        type: 'number',
        label: 'What is your age?',
        required: true,
        min: 18,
        max: 100
    },
    {
        name: 'gender',
        type: 'radio',
        label: 'What is your gender?',
        required: true,
        choices: [
            'Male',
            'Female', 
            'Other',
            'Prefer not to say'
        ]
    },
    {
        name: 'education',
        type: 'select',
        label: 'What is your highest level of education?',
        required: true,
        choices: [
            'High school or less',
            'Some college',
            'Bachelor\'s degree',
            'Master\'s degree',
            'PhD or higher',
            'Other'
        ]
    },
    {
        name: 'english_fluency',
        type: 'radio',
        label: 'How would you rate your English fluency?',
        required: true,
        choices: [
            'Native speaker',
            'Fluent',
            'Good',
            'Fair',
            'Poor'
        ]
    },
    {
        name: 'country',
        type: 'text',
        label: 'What country do you currently live in?',
        required: true
    },
    {
        name: 'participation_before',
        type: 'radio',
        label: 'Have you participated in a similar study before?',
        required: true,
        choices: [
            'Yes',
            'No',
            'Not sure'
        ]
    },
    {
        name: 'feedback',
        type: 'textarea',
        label: 'Do you have any feedback about this study? (Optional)',
        required: false,
        rows: 4
    }
];

// Function to generate form HTML
function generateDemographicsForm() {
    let html = '<div class="demographics-form">';
    
    window.demographicsQuestions.forEach(function(question) {
        html += '<div class="form-group mb-3">';
        html += '<label class="form-label">' + question.label;
        if (question.required) {
            html += ' <span class="text-danger">*</span>';
        }
        html += '</label>';
        
        if (question.type === 'text') {
            html += '<input type="text" class="form-control" name="' + question.name + '"';
            if (question.required) html += ' required';
            html += '>';
        }
        else if (question.type === 'number') {
            html += '<input type="number" class="form-control" name="' + question.name + '"';
            if (question.min) html += ' min="' + question.min + '"';
            if (question.max) html += ' max="' + question.max + '"';
            if (question.required) html += ' required';
            html += '>';
        }
        else if (question.type === 'textarea') {
            html += '<textarea class="form-control" name="' + question.name + '"';
            if (question.rows) html += ' rows="' + question.rows + '"';
            if (question.required) html += ' required';
            html += '></textarea>';
        }
        else if (question.type === 'radio') {
            question.choices.forEach(function(choice, index) {
                html += '<div class="form-check">';
                html += '<input class="form-check-input" type="radio" name="' + question.name + '" id="' + question.name + '_' + index + '" value="' + choice + '"';
                if (question.required) html += ' required';
                html += '>';
                html += '<label class="form-check-label" for="' + question.name + '_' + index + '">' + choice + '</label>';
                html += '</div>';
            });
        }
        else if (question.type === 'select') {
            html += '<select class="form-control" name="' + question.name + '"';
            if (question.required) html += ' required';
            html += '>';
            html += '<option value="">Please select...</option>';
            question.choices.forEach(function(choice) {
                html += '<option value="' + choice + '">' + choice + '</option>';
            });
            html += '</select>';
        }
        
        html += '</div>';
    });
    
    html += '</div>';
    return html;
}

// Initialize demographics form when document is ready
document.addEventListener('DOMContentLoaded', function() {
    const demographicsContainer = document.getElementById('demographics-container');
    if (demographicsContainer) {
        demographicsContainer.innerHTML = generateDemographicsForm();
    }
});
