from hybrid_medical_agent.agent import controller


def run():
    payload = {"side_effects": {"en": None, "fr": None}}
    result = controller.Controller._extract_field(payload, "side_effects")
    print("Extracted:", repr(result))


if __name__ == '__main__':
    run()
