import emergent
from emergent import ChatAgent
import json


AVAILABLE = {
        "lee": {
            "Monday": ["10:00", "11:00", "12:00"],
            "Tuesday": ["10:00", "11:00", "12:00"],
            "Wednesday": ["10:00", "11:00", "12:00"],
        },
        "smith": {
            "Friday": ["16:00", "17:00", "18:00"],
            "Saturday": ["10:00", "11:00", "12:00"],
            "Sunday": ["10:00", "11:00", "12:00"],
        },
    }

@emergent.tool()
def available_times(
    doctor: "Name of doctor (categorical) one of either [lee, smith]",
):
    """Used to request the available times for a given doctor."""

    

    doctor = doctor.lower()

    if any(x in doctor for x in ["avery", "lee"]):
        doctor = "lee"
    elif any(x in doctor for x in ["smith", "james"]):
        doctor = "smith"
    else:
        result = f"\n\nError! doctor '{doctor}' not found. "
        result += "Try again and choose from [Dr Lee, Dr Smith.]"
        return result

    times = AVAILABLE[doctor]

    result = "\n\n\nThe available times for Dr. {} are:\n".format(
        doctor.title()
    )

    for day, times in times.items():
        result += "- {}: {}\n".format(day, ", ".join(times))

    # result += "Remember, the user cannot see this information, so you might "
    # result += "need to replicate this information in your response when needed."

    return result


@emergent.tool()
def book_appointment(
    doctor: "Name of doctor",
    time: "Time of appointment",
    date: "Date of appointment",
    full_name: "Full name of patient",
    age: "Age of patient",
    phone_number: "Phone number of patient",
    reason: "Reason for appointment",
):
    """This tool must be used to book an appointment with a doctor."""

    return "Appointment booked SUCCESSFULLY. You should confirm with the user now."


class SearchEngine:

    @emergent.tool()
    def search(self, query):
        """This tool is useful for searching the web"""

        return "obamas current age is 83"


engine = SearchEngine()

print(engine.search("hi"))

agent = ChatAgent(
    tools=[available_times, book_appointment, engine.search], model="gpt-4"
)



print(agent.system_prompt)

try:
    while True:
        message = input("You: ")
        if message == "quit":
            print(json.dumps(agent.messages, indent=4))
            break
        print()
        print("Assistant:", agent.send(message))
        print()
except:
    print(json.dumps(agent.messages, indent=4))
    import traceback
    traceback.print_exc()