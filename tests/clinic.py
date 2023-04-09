import emergent
from emergent import ChatAgent
import json


AVAILABLE = {
    "lee": {
        "Monday": ["10am", "11am", "12pm"],
        "Tuesday": ["10am", "11am", "12pm"],
        "Wednesday": ["10am", "11am", "12pm"],
    },
    "smith": {
        "Friday": ["4pm", "5pm", "6pm"],
        "Saturday": ["10am", "11am", "12pm"],
        "Sunday": ["10am", "11am", "12pm"],
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

    result = "\n\n\nThe available times for Dr. {} are:\n".format(doctor.title())

    for day, times in times.items():
        result += "- {}: {}\n".format(day, ", ".join(times))

    # result += "Remember, the user cannot see this information, so you might "
    # result += "need to replicate this information in your response when needed."

    return result


@emergent.tool()
def book_appointment(
    doctor: "Name of doctor",
    time: "Time of appointment",
    day: "day of appointment",
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

        return "obamas current age is 63"


engine = SearchEngine()

agent = ChatAgent(
    tools=[available_times, book_appointment, engine.search], model="gpt-4"
)

agent.personality = """
You are an extremely kind and friendly AI assistant. 
Your primary purpose is to help patients book appointments for our clinic (Greenstar Medical Center). 
You were developed by duos.ai. 

Here is some of the clinics information:
- We have two doctors: Dr. Avery Lee and Dr. John Smith
- We are located at 123 Main St, New York, NY 10001
- Our phone number is (212) 555-1234
- Our website is greenstarmedical.com
"""


agent.run()
