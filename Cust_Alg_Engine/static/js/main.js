$(document).ready(function () {
	
	const form = document.querySelector('#form');
    const checkboxes = form.querySelectorAll('input[class=features]');
    const checkboxLength = checkboxes.length;
    const firstCheckbox = checkboxLength > 0 ? checkboxes[0] : null;
	
	//
	function init() {
        if (firstCheckbox) {
            for (let i = 0; i < checkboxLength; i++) {
                checkboxes[i].addEventListener('change', checkValidity);
            }

            checkValidity();
        }
    }
	
	//
	 function isChecked() {
        for (let i = 0; i < checkboxLength; i++) {
            if (checkboxes[i].checked) return true;
        }

        return false;
    }

	//
    function checkValidity() {
        const errorMessage = !isChecked() ? 'At least one checkbox must be selected.' : '';
        firstCheckbox.setCustomValidity(errorMessage);
    }	

	//to initialize the init function
    init();
	});
	
	
	//select all toggle function
	function toggle1(source) {
    var checkboxes = document.getElementsByName('selected_categorical_columns');
    for (var i = 0; i < checkboxes.length; i++) {
        if (checkboxes[i] != source)
            checkboxes[i].checked = source.checked;
    }
}
	function toggle2(source) {
    var checkboxes = document.getElementsByName('selected_numerical_columns');
    for (var i = 0; i < checkboxes.length; i++) {
        if (checkboxes[i] != source)
            checkboxes[i].checked = source.checked;
    }
}
	
	
	//For disbaling the start button afetr user clicks on it
    function disableButton() {
        var btn = document.getElementById('upload');
        btn.disabled = true;
        btn.innerText = 'Processing...'
    }