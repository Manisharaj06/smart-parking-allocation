from flask import Flask, render_template, redirect, url_for
import numpy as np
from env.parking_env import ParkingEnv

app = Flask(__name__)

# Load environment
env = ParkingEnv()

# Load trained Q-table
q_table = np.load("models/q_table.npy")

# Random initial parking state
current_state = np.random.randint(
    0,
    2,
    size=env.total_slots
)

# Dashboard variables
total_vehicles = 0
last_reward = 0
last_selected_slot = None
allocation_status = "Waiting for incoming vehicle..."

# Vehicle queue
vehicle_queue = 3


@app.route("/")
def home():

    global current_state
    global total_vehicles
    global last_reward
    global last_selected_slot
    global allocation_status
    global vehicle_queue

    slots = []

    free_slots = 0
    occupied_slots = 0

    # Predict next best slot
    selected_slot = None

    if np.any(current_state == 0):

        state_index = int(
            "".join(map(str, current_state)),
            2
        )

        selected_slot = np.argmax(
            q_table[state_index]
        )

    # Generate slot UI
    for i, value in enumerate(current_state):

        if i == selected_slot and value == 0:

            slots.append("selected")
            free_slots += 1

        elif value == 0:

            slots.append("free")
            free_slots += 1

        else:

            slots.append("occupied")
            occupied_slots += 1

    occupancy = int(
        (occupied_slots / env.total_slots) * 100
    )

    return render_template(
        "index.html",
        slots=slots,
        occupancy=occupancy,
        free_slots=free_slots,
        occupied_slots=occupied_slots,
        total_vehicles=total_vehicles,
        last_reward=last_reward,
        last_selected_slot=last_selected_slot,
        allocation_status=allocation_status,
        parking_state=current_state.tolist(),
        vehicle_queue=vehicle_queue
    )


@app.route("/allocate")
def allocate():

    global current_state
    global total_vehicles
    global last_reward
    global last_selected_slot
    global allocation_status
    global vehicle_queue

    # Check if parking full
    if np.all(current_state == 1):

        allocation_status = "Parking lot is full!"

        return redirect(url_for("home"))

    # Convert state to index
    state_index = int(
        "".join(map(str, current_state)),
        2
    )

    # Exploration vs exploitation
    epsilon = np.random.uniform(0, 1)

    # 40% exploration
    if epsilon < 0.4:

        action = np.random.randint(
            0,
            env.total_slots
        )

        decision_type = "Exploration"

    # 60% exploitation
    else:

        # Small randomness even in AI decisions
        if np.random.rand() < 0.3:

            available_actions = np.where(
                current_state == 0
            )[0]

            action = np.random.choice(
                available_actions
            )

        else:

            action = np.argmax(
                q_table[state_index]
            )

        decision_type = "Exploitation"

    last_selected_slot = f"P{action + 1}"

    # Correct allocation
    if current_state[action] == 0:

        current_state[action] = 1

        # Variable positive rewards
        last_reward = np.random.choice(
            [5, 10, 15]
        )

        allocation_status = (
            f"{decision_type}: Vehicle allocated successfully to P{action + 1}"
        )

        total_vehicles += 1

        # Update queue dynamically
        vehicle_queue = max(
            0,
            vehicle_queue - 1
        )

        # Random new incoming vehicles
        vehicle_queue += np.random.randint(1, 3)

    # Wrong allocation
    else:

        # Variable negative rewards
        last_reward = np.random.choice(
            [-5, -10, -15]
        )

        allocation_status = (
            f"{decision_type}: Wrong allocation! "
            f"P{action + 1} already occupied."
        )

    return redirect(url_for("home"))


@app.route("/reset")
def reset():

    global current_state
    global total_vehicles
    global last_reward
    global last_selected_slot
    global allocation_status
    global vehicle_queue

    # Random parking initialization
    current_state = np.random.randint(
        0,
        2,
        size=env.total_slots
    )

    total_vehicles = 0

    last_reward = 0

    last_selected_slot = None

    # Random queue size
    vehicle_queue = np.random.randint(2, 5)

    allocation_status = (
        "Parking environment reset successfully."
    )

    return redirect(url_for("home"))


if __name__ == "__main__":

    app.run(debug=True,port=5001)