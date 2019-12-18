## Scikitlearn
We created a maxent and laplace model using scikitlearn. These models were well optimized and far better able to classify data than our own models.
F1 scores for both models are .999 or higher using English, French, German, and Italian on the LEPZIG corpus. 
However, we were unable to figure out a way to use scikitlearn to classify for two classes which have no priority. 
I.e. if we have a sentence that is Spanish and English and we do not care if the system thinks Spanish or English is more likely as long as we identify both languages,
we would need a system that is able to reflect that result. We were neither able to utilize scikit for predicting two most likely languages nor were we able
to find a way to ignore the order of the class predictions. It may be the case that scikit learn or some other module could accomplish the results we needed. 
We were unable to persue a good multiclass bilingual code switching maxent model because we were too busy procuring data. We were however very successful with creating
various models using both modules and our own code to predict single language class. 

No good corpus for code-switching multilngual sentences in the French, English, German, Spanish, Italian, and Dutch languages was found. Our analysis of
twitter data was only partially successful due to limits to our free developer twitter api access as well as limits in resources for annotating and procuring data.

#### To run our scikit learn laplace and maxent language classifiers, one must be in the folder "scikitLearnMaxentLaplace" and run the following command:
python project_code.py