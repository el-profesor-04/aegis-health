from graph.schema import HealthGraph
from ingestion.pipeline import ingest_text
from visualization.graph_viz import visualize_graph


# =========================
# TEST DATA (10 USERS)
# =========================

users = {

    "user_1_runner_knee": [
        "My left knee has been acting up again after yesterday’s run",
        "Sharp pain when I go downstairs, mostly in the knee cap area",
        "I think it started after I increased my mileage last week",
        "Slight swelling but only after long runs",
        "Took ibuprofen but not sure it helped",
        "Today it feels stiff even when sitting",
        "I ran through the pain again this morning",
        "Pain is worse when cold outside",
        "Not sure if it’s tendon or joint issue",
        "I might need to stop running for a bit"
    ],

    "user_2_gut": [
        "I felt fine in the morning but after coffee my stomach felt weird",
        "Maybe the sandwich I had was bad, not sure",
        "Nausea came and went throughout the day",
        "Mild cramps after lunch",
        "I also had bloating after dinner",
        "Could be lactose again but I’m not sure",
        "Felt dizzy in the evening",
        "No vomiting but close",
        "Stress might be making it worse",
        "I skipped breakfast yesterday too"
    ],

    "user_3_gym": [
        "Did heavy chest workout yesterday, now shoulders hurt",
        "Right shoulder feels sharper than left",
        "I also feel soreness in triceps and upper back",
        "Might have overdone bench press",
        "Pain only starts when lifting arm above head",
        "Took protein shake and caffeine pre-workout",
        "Sleep was bad last night",
        "Feeling generally fatigued today",
        "Not sure if it's injury or DOMS",
        "Still planning to train today"
    ],

    "user_4_stress": [
        "I haven’t been sleeping well for the past week",
        "Heart feels like it’s racing sometimes at night",
        "Random chest tightness but it goes away",
        "I feel anxious without reason",
        "Drinking more coffee than usual",
        "Headaches in the afternoon",
        "Hard to focus at work",
        "Sometimes I feel short of breath but it passes",
        "Could be stress or something physical",
        "I skipped gym all week"
    ],

    "user_5_compensation": [
        "Hurt my right ankle playing basketball",
        "Now my left knee is starting to hurt too",
        "I think I’m walking differently",
        "Swelling in ankle reduced but still weak",
        "Knee pain shows up when climbing stairs",
        "Back feels slightly tight as well",
        "I’m limping a bit without noticing",
        "Pain is not constant, only during movement",
        "Took rest for 2 days",
        "Still want to play again soon"
    ],

    "user_6_flu": [
        "Started feeling feverish yesterday evening",
        "Body aches all over, especially back and legs",
        "Mild sore throat today",
        "Not sure if it’s flu or just fatigue",
        "Took paracetamol",
        "Slept most of the afternoon",
        "No cough yet but throat feels scratchy",
        "Loss of appetite",
        "Drinking less water than usual",
        "Might go to work tomorrow anyway"
    ],

    "user_7_lifestyle": [
        "I think my diet has been bad lately",
        "Sometimes I eat once a day, sometimes twice",
        "Energy crashes in the afternoon",
        "Head feels foggy after lunch",
        "I started intermittent fasting but not consistent",
        "Sometimes I drink energy drinks instead of meals",
        "Weight feels like it’s fluctuating",
        "No exercise this week",
        "I don’t know what’s causing what anymore",
        "Sleep schedule is all over the place"
    ],

    "user_8_posture": [
        "My neck hurts when I wake up",
        "Lower back stiffness during the day",
        "Shoulder tension from sitting at desk",
        "Sometimes pain shifts between upper back and neck",
        "I work long hours on laptop",
        "Stretching helps temporarily",
        "No injury that I remember",
        "Could be posture issue",
        "Pain increases after meetings",
        "Heat pack seems to help a bit"
    ],

    "user_9_medication": [
        "Started new allergy medication last week",
        "Now feeling drowsy during the day",
        "Dry mouth is constant",
        "Slight headaches in mornings",
        "Not sure if medication or dehydration",
        "Symptoms started after dosage increase",
        "Allergies themselves are better though",
        "Eyes feel less itchy",
        "Hard to concentrate at work",
        "Thinking of stopping medication"
    ],

    "user_10_chronic_mix": [
        "My knee pain from last year is back",
        "But this time it feels different",
        "Also started getting hip discomfort",
        "Could be from sitting too long again",
        "Ran a short distance and pain spiked",
        "Not sure if old injury or new one",
        "Sometimes no pain at all during day",
        "Pain returns randomly at night",
        "I might have ignored early signs",
        "Feeling frustrated because it keeps coming back"
    ]
}


# =========================
# RUN TESTS
# =========================

def run_user_test(user_id, inputs):
    print(f"\n\n==================== {user_id} ====================")

    graph = HealthGraph()

    for text in inputs:
        ingest_text(graph, text)

    # visualize per user
    output_file = f"graph_{user_id}.html"
    visualize_graph(graph, output_file)

    print(f"Saved graph: {output_file}")


if __name__ == "__main__":
    for user_id, inputs in users.items():
        run_user_test(user_id, inputs)