from hybrid_medical_agent.agent.training_brain import TrainingBrain


def run_test():
    tb = TrainingBrain()
    knowledge = '{"composition": "benzalkonium chloride 0.18%"}'
    product = "BACTOL"
    user_claim = "BACTOL contains paracetamol"

    for level in [1, 2, 3, 4]:
        decision = tb.decide_next_action(
            conversation_history=[],
            conversation_state={"current_product": product, "product_locked": True},
            memory_context={},
            detected_product=product,
            alia_level=level,
            last_topic=None,
            knowledge_context=knowledge,
            user_message=user_claim,
        )
        print(f"ALIA level {level} -> action: {decision.action}, message: {decision.message}\n")


if __name__ == "__main__":
    run_test()
