

function addItem() {
    if (this.items.length < this.maxItems) {
        // we set sense insestive flag to window.caseflag. if it is not set, we set it to true (default)

        const bool_caseflag = window.caseflag === undefined ? true : window.caseflag;
        const caseflag = bool_caseflag ? 'i' : '';


        let fields = window.suffixes.map((suffix, index) => ({
            value: '',
            suffix: suffix,
            regex: new RegExp(window.regex[index], caseflag), // Add regex here
            allowedValues: window.allowedValues[index].map(item => item.toLowerCase())
        }));
        this.items.push({fields: fields});
    }
}

function getFields() {
    const bool_caseflag = window.caseflag === undefined ? true : window.caseflag;
    const caseflag = bool_caseflag ? 'i' : '';
    let fields = window.suffixes.map((suffix, index) => ({
        value: '',
        suffix: suffix,
        regex: new RegExp(window.regex[index], caseflag),
        allowedValues: window.allowedValues[index].map(item => item.toLowerCase())
    }));
    return fields
}

function checkSingleField(field) {
    const convValue = field.value.trim().replace(/\s+/g, ' ');
    const regex = field.regex;
    console.debug(convValue, regex, regex.test(convValue))
    console.debug('*')
    return regex.test(convValue);
}