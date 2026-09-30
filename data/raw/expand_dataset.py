import json
from pathlib import Path


OUTPUT = Path("data/raw/medical_qa.jsonl")


DATA = [
    # Anatomy
    ("What is the structure of a nephron?", "A nephron is the functional unit of the kidney consisting of a renal corpuscle and renal tubule."),
    ("What is the glomerulus?", "The glomerulus is a network of capillaries within the renal corpuscle where blood filtration begins."),
    ("What is the function of the stomach?", "The stomach stores food, mixes it with gastric secretions, and begins digestion, particularly of proteins."),
    ("What is the function of the small intestine?", "The small intestine is the main site for digestion and absorption of nutrients."),
    ("What is the function of the large intestine?", "The large intestine absorbs water and electrolytes and helps form and store feces."),
    ("What is the function of the liver?", "The liver performs metabolic functions, produces bile, stores nutrients, detoxifies substances, and synthesizes plasma proteins."),
    ("What is the function of the spleen?", "The spleen filters blood, removes aged blood cells, and participates in immune responses."),
    ("What is the function of the heart?", "The heart pumps blood through the pulmonary and systemic circulations."),
    ("What is the function of arteries?", "Arteries carry blood away from the heart toward tissues and organs."),
    ("What is the function of veins?", "Veins return blood from tissues to the heart."),
    ("What are capillaries?", "Capillaries are small blood vessels where exchange of gases, nutrients, and waste products occurs between blood and tissues."),
    ("What is the function of the lungs?", "The lungs exchange oxygen and carbon dioxide between inhaled air and blood."),
    ("What is the diaphragm?", "The diaphragm is a major muscle of respiration that contracts to help draw air into the lungs."),
    ("What is the function of the spinal cord?", "The spinal cord transmits signals between the brain and body and coordinates many reflexes."),
    ("What is the cerebral cortex?", "The cerebral cortex is the outer layer of the brain involved in functions such as perception, movement, language, and cognition."),
    ("What is the function of the cerebellum?", "The cerebellum helps coordinate movement, balance, posture, and motor learning."),
    ("What is the function of the pancreas?", "The pancreas produces digestive enzymes and hormones such as insulin and glucagon."),
    ("What is the function of the thyroid gland?", "The thyroid produces hormones that regulate metabolism, growth, and development."),
    ("What is the function of the adrenal glands?", "The adrenal glands produce hormones including cortisol, aldosterone, and adrenaline."),
    ("What is the function of bone marrow?", "Bone marrow produces blood cells through the process of hematopoiesis."),

    # Physiology
    ("What is homeostasis?", "Homeostasis is the maintenance of relatively stable internal conditions despite changes in the external environment."),
    ("What is blood pressure?", "Blood pressure is the force exerted by circulating blood against the walls of blood vessels."),
    ("What is cardiac output?", "Cardiac output is the volume of blood pumped by the heart per minute."),
    ("What is heart rate?", "Heart rate is the number of heartbeats occurring per minute."),
    ("What is stroke volume?", "Stroke volume is the amount of blood ejected by a ventricle during one heartbeat."),
    ("What is respiration?", "Respiration includes the processes involved in taking in oxygen and eliminating carbon dioxide."),
    ("What is gas exchange?", "Gas exchange is the movement of oxygen and carbon dioxide between air and blood or between blood and tissues."),
    ("What is filtration in the kidney?", "Renal filtration is the process by which water and small dissolved substances pass from glomerular blood into the nephron."),
    ("What is tubular reabsorption?", "Tubular reabsorption is the movement of useful substances and water from the renal tubule back into the blood."),
    ("What is tubular secretion?", "Tubular secretion is the movement of selected substances from blood into the renal tubule for elimination."),
    ("What is insulin?", "Insulin is a hormone that lowers blood glucose by promoting glucose uptake and storage and reducing hepatic glucose production."),
    ("What is glucagon?", "Glucagon is a hormone that raises blood glucose, partly by stimulating hepatic glucose production."),
    ("What is body temperature regulation?", "Body temperature regulation maintains core temperature through coordinated nervous, hormonal, vascular, and metabolic responses."),
    ("What is a reflex?", "A reflex is a rapid, automatic response to a stimulus mediated through a neural pathway."),
    ("What is an action potential?", "An action potential is a rapid change in membrane potential that propagates along an excitable cell."),
    ("What is resting membrane potential?", "Resting membrane potential is the electrical potential difference across a cell membrane when the cell is not generating an action potential."),
    ("What is synaptic transmission?", "Synaptic transmission is communication between neurons or between a neuron and an effector cell across a synapse."),
    ("What is oxygen saturation?", "Oxygen saturation is the proportion of hemoglobin binding sites occupied by oxygen."),
    ("What is acid-base balance?", "Acid-base balance is the regulation of hydrogen ion concentration to maintain a suitable pH in body fluids."),
    ("What is erythropoiesis?", "Erythropoiesis is the production of red blood cells, primarily in the bone marrow in adults."),

    # Biochemistry
    ("What is ATP?", "ATP is the main immediate energy-carrying molecule used by cells."),
    ("What is glucose?", "Glucose is a simple carbohydrate that serves as an important energy source for cells."),
    ("What is glycolysis?", "Glycolysis is the metabolic pathway that breaks down glucose into pyruvate while generating ATP and reducing equivalents."),
    ("What is the citric acid cycle?", "The citric acid cycle is a metabolic pathway that oxidizes acetyl-CoA and generates reducing equivalents used for ATP production."),
    ("What is oxidative phosphorylation?", "Oxidative phosphorylation uses an electron transport chain and proton gradient to generate ATP."),
    ("What are proteins?", "Proteins are polymers of amino acids that perform structural, enzymatic, transport, signaling, and other functions."),
    ("What are amino acids?", "Amino acids are organic molecules that serve as the building blocks of proteins."),
    ("What are lipids?", "Lipids are hydrophobic or amphipathic molecules involved in energy storage, membranes, and signaling."),
    ("What is DNA?", "DNA stores genetic information and provides instructions used for the development and function of organisms."),
    ("What is RNA?", "RNA is a nucleic acid involved in processes including gene expression and protein synthesis."),
    ("What is an enzyme?", "An enzyme is a biological catalyst that increases the rate of a chemical reaction without being consumed."),
    ("What is an active site?", "The active site is the region of an enzyme where substrate binding and catalysis occur."),
    ("What is metabolism?", "Metabolism is the collection of chemical reactions that maintain cellular life and energy balance."),
    ("What is glycogen?", "Glycogen is a stored form of glucose found mainly in the liver and skeletal muscle."),
    ("What is cholesterol?", "Cholesterol is a lipid that contributes to cell membranes and serves as a precursor for steroid hormones and bile acids."),
    ("What is hemoglobin?", "Hemoglobin is a protein in red blood cells that primarily transports oxygen."),
    ("What is an electrolyte?", "An electrolyte is a substance that forms ions in solution and contributes to electrical and physiological processes."),
    ("What is pH?", "pH is a measure related to the hydrogen ion concentration of a solution."),
    ("What is a nucleotide?", "A nucleotide consists of a nitrogenous base, a sugar, and one or more phosphate groups."),
    ("What is protein synthesis?", "Protein synthesis is the cellular process of producing proteins using genetic information encoded in nucleic acids."),

    # Pathology
    ("What is anemia?", "Anemia is a condition in which blood has reduced oxygen-carrying capacity, commonly associated with reduced hemoglobin."),
    ("What is hypertension?", "Hypertension is persistently elevated blood pressure that increases long-term cardiovascular and other health risks."),
    ("What is diabetes mellitus?", "Diabetes mellitus is a group of metabolic disorders characterized by chronically elevated blood glucose."),
    ("What is inflammation?", "Inflammation is a biological response to tissue injury or harmful stimuli that helps protect and repair tissue."),
    ("What is infection?", "An infection occurs when microorganisms enter and multiply within a host and may cause tissue injury or disease."),
    ("What is fever?", "Fever is an elevation of regulated body temperature commonly associated with an immune response to infection or inflammation."),
    ("What is thrombosis?", "Thrombosis is the formation of a blood clot within a blood vessel."),
    ("What is edema?", "Edema is abnormal accumulation of fluid in the tissues."),
    ("What is ischemia?", "Ischemia is inadequate blood flow to a tissue, resulting in reduced oxygen and nutrient delivery."),
    ("What is hypoxia?", "Hypoxia is a state in which tissues receive insufficient oxygen."),
    ("What is atherosclerosis?", "Atherosclerosis is the buildup of lipid-rich plaques within arterial walls."),
    ("What is pneumonia?", "Pneumonia is an infection or inflammatory condition of lung tissue that can impair gas exchange."),
    ("What is asthma?", "Asthma is a chronic inflammatory airway disorder characterized by variable airflow limitation and airway hyperresponsiveness."),
    ("What is osteoporosis?", "Osteoporosis is a disorder characterized by reduced bone strength and increased fracture risk."),
    ("What is arthritis?", "Arthritis refers to conditions involving inflammation or structural disease of joints."),
    ("What is sepsis?", "Sepsis is life-threatening organ dysfunction caused by a dysregulated response to infection."),
    ("What is a stroke?", "A stroke occurs when blood flow to part of the brain is interrupted or when a blood vessel in the brain ruptures."),
    ("What is myocardial infarction?", "Myocardial infarction occurs when prolonged interruption of blood flow causes injury to heart muscle."),
    ("What is renal failure?", "Renal failure is severe loss of kidney function that impairs waste removal and regulation of fluid and electrolytes."),
    ("What is dehydration?", "Dehydration occurs when the body loses more water than it takes in, resulting in a deficit of body fluid."),

    # Pharmacology
    ("What is pharmacology?", "Pharmacology is the study of drugs and their effects on living organisms."),
    ("What is pharmacokinetics?", "Pharmacokinetics describes how the body absorbs, distributes, metabolizes, and eliminates a drug."),
    ("What is pharmacodynamics?", "Pharmacodynamics describes the effects of a drug on the body and the mechanisms producing those effects."),
    ("What is an agonist?", "An agonist is a substance that binds to a receptor and activates it."),
    ("What is an antagonist?", "An antagonist binds to a receptor and blocks or reduces the action of an agonist."),
    ("What is drug absorption?", "Drug absorption is the movement of a drug from its site of administration into the systemic circulation."),
    ("What is drug distribution?", "Drug distribution is the movement of a drug from the bloodstream into tissues and body compartments."),
    ("What is drug metabolism?", "Drug metabolism is the chemical modification of drugs, often occurring in the liver."),
    ("What is drug excretion?", "Drug excretion is the removal of drugs or their metabolites from the body, commonly through the kidneys."),
    ("What is a side effect?", "A side effect is an unintended effect that occurs in addition to a drug's intended therapeutic effect."),
    ("What is an adverse drug reaction?", "An adverse drug reaction is an unintended and harmful response to a drug used at normal doses."),
    ("What is an antibiotic?", "An antibiotic is a drug used to treat infections caused by susceptible bacteria."),
    ("What is analgesia?", "Analgesia is the reduction or absence of pain without necessarily causing loss of consciousness."),
    ("What is an antihypertensive?", "An antihypertensive is a drug used to reduce elevated blood pressure."),
    ("What is insulin therapy?", "Insulin therapy uses administered insulin to help control blood glucose when endogenous insulin is insufficient or ineffective."),

    # Immunology
    ("What is the immune system?", "The immune system is a network of cells, tissues, organs, and molecules that identifies and responds to pathogens and abnormal cells."),
    ("What are antibodies?", "Antibodies are proteins produced by B cells that specifically bind to antigens."),
    ("What are B cells?", "B cells are lymphocytes involved in adaptive immunity and can differentiate into antibody-producing plasma cells."),
    ("What are T cells?", "T cells are lymphocytes that perform several roles in adaptive immunity, including immune regulation and killing infected cells."),
    ("What is an antigen?", "An antigen is a substance that can be recognized by components of the immune system."),
    ("What is innate immunity?", "Innate immunity is the rapid, non-specific defense system present from birth."),
    ("What is adaptive immunity?", "Adaptive immunity is an antigen-specific immune response that develops through exposure and can generate immunological memory."),
    ("What is immunological memory?", "Immunological memory is the ability of the adaptive immune system to respond more rapidly to a previously encountered antigen."),
    ("What is an autoimmune disease?", "An autoimmune disease occurs when the immune system mistakenly attacks the body's own tissues."),
    ("What is vaccination?", "Vaccination stimulates an immune response that can provide protection against a specific infectious disease."),

    # Clinical/basic medical
    ("What is a pulse oximeter?", "A pulse oximeter is a noninvasive device commonly used to estimate blood oxygen saturation and pulse rate."),
    ("What is a blood test?", "A blood test analyzes blood components or biomarkers to help assess health and diagnose or monitor disease."),
    ("What is hemoglobin concentration?", "Hemoglobin concentration measures the amount of hemoglobin present in a given volume of blood."),
    ("What is a complete blood count?", "A complete blood count measures major cellular components of blood, including red cells, white cells, and platelets."),
    ("What are platelets?", "Platelets are blood components that help form clots and contribute to hemostasis after vascular injury."),
    ("What are white blood cells?", "White blood cells are immune cells that participate in defense against pathogens and abnormal cells."),
    ("What are red blood cells?", "Red blood cells transport oxygen from the lungs to tissues and help transport carbon dioxide."),
    ("What is a blood clot?", "A blood clot is a mass of blood components formed during coagulation to reduce bleeding."),
    ("What is heart rate variability?", "Heart rate variability describes variation in the time intervals between successive heartbeats."),
    ("What is an ECG?", "An electrocardiogram records the electrical activity of the heart over time."),
    ("What is an MRI?", "Magnetic resonance imaging uses magnetic fields and radio waves to produce detailed images of internal body structures."),
    ("What is a CT scan?", "Computed tomography uses X-rays and computer processing to produce cross-sectional images of the body."),
    ("What is an ultrasound?", "Ultrasound uses high-frequency sound waves to produce images of internal structures."),
    ("What is a biopsy?", "A biopsy involves removing a sample of tissue for microscopic or laboratory examination."),
    ("What is a symptom?", "A symptom is a subjective experience reported by a patient, such as pain or nausea."),
    ("What is a sign in medicine?", "A medical sign is an objective finding that can be observed or measured during examination or testing."),
    ("What is diagnosis?", "Diagnosis is the process of identifying a disease or condition based on clinical information and investigations."),
    ("What is prognosis?", "Prognosis is an assessment of the expected course or outcome of a disease or condition."),
    ("What is a clinical risk factor?", "A risk factor is a characteristic or exposure associated with an increased likelihood of developing a disease or health outcome."),
    ("What is preventive medicine?", "Preventive medicine focuses on reducing disease risk and maintaining health through interventions such as screening and vaccination."),

    # Additional core concepts
    ("What is cell division?", "Cell division is the process by which a cell produces daughter cells."),
    ("What is mitosis?", "Mitosis is a type of cell division that produces genetically similar daughter cells."),
    ("What is meiosis?", "Meiosis is a specialized cell division that produces reproductive cells with half the usual chromosome number."),
    ("What is apoptosis?", "Apoptosis is a regulated process of programmed cell death."),
    ("What is a chromosome?", "A chromosome is a DNA-containing structure that carries genetic information."),
    ("What is a gene?", "A gene is a segment of genetic material that contains information contributing to a functional product."),
    ("What is gene expression?", "Gene expression is the process by which information encoded in a gene is used to produce a functional product."),
    ("What is transcription?", "Transcription is the synthesis of RNA using DNA as a template."),
    ("What is translation?", "Translation is the synthesis of a protein using the information carried by messenger RNA."),
    ("What is a mutation?", "A mutation is a change in the nucleotide sequence of genetic material."),
    ("What is a hormone?", "A hormone is a signaling molecule produced by cells or glands that acts on target cells."),
    ("What is a receptor?", "A receptor is a protein or other molecular structure that recognizes a signaling molecule and initiates a cellular response."),
    ("What is a neurotransmitter?", "A neurotransmitter is a chemical messenger released by neurons to communicate with other cells."),
    ("What is a neuron?", "A neuron is a specialized cell that receives, processes, and transmits electrical and chemical signals."),
    ("What is a synapse?", "A synapse is a specialized junction through which a neuron communicates with another cell."),
    ("What is tissue?", "Tissue is an organized group of similar cells and associated components performing specific functions."),
    ("What is an organ?", "An organ is a structure composed of multiple tissue types that work together to perform specific functions."),
    ("What is an organ system?", "An organ system is a group of organs that cooperate to perform major physiological functions."),
    ("What is extracellular fluid?", "Extracellular fluid is the fluid outside cells that provides the surrounding environment for cellular activity."),
    ("What is intracellular fluid?", "Intracellular fluid is the fluid contained within cells and makes up a major portion of total body water."),
]


def main():
    with OUTPUT.open("w", encoding="utf-8") as f:
        for question, answer in DATA:
            f.write(
                json.dumps(
                    {
                        "instruction": question,
                        "answer": answer,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Wrote {len(DATA)} records to {OUTPUT}")


if __name__ == "__main__":
    main()