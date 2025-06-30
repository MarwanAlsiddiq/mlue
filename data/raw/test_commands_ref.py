import unittest
from dragonfly import RecognitionFailed, Grammar, CompoundRule, RuleRef
from breathe.elements.commands_ref import CommandsRef, Exec

# Mock Action for testing
class MockAction:
    def __init__(self):
        self.executed = False

    def execute(self):
        self.executed = True

class TestCommandsRef(unittest.TestCase):

    def test_repetition(self):
        ref = CommandsRef("test_ref", max=2, min=1)
        self.assertEqual(ref.min, 1)
        self.assertEqual(ref.max, 3)  # max is internally incremented by 1
        self.assertEqual(ref.name, "test_ref")


class TestExec(unittest.TestCase):

    def test_execute(self):
        exec_action = Exec("test_exec")
        action1 = MockAction()
        action2 = MockAction()
        data = {"test_exec": [action1, action2]}
        exec_action._execute(data)
        self.assertTrue(action1.executed)
        self.assertTrue(action2.executed)

    def test_execute_empty(self):
        exec_action = Exec("test_exec")
        data = {} # Key not present
        exec_action._execute(data)  # Should not raise an error

    def test_execute_not_list(self):
        exec_action = Exec("test_exec")
        data = {"test_exec": "not a list"}
        with self.assertRaises(TypeError):
            exec_action._execute(data)


    def test_integration_with_grammar(self): # Demonstrates usage within a grammar
        action1 = MockAction()
        action2 = MockAction()

        class TestRule(CompoundRule):
            spec = "test <test_ref>"
            extras = [
                CommandsRef("test_ref", max=2, min=1),
                RuleRef(rule=action1, name="action1"),
                RuleRef(rule=action2, name="action2"),
            ]
            exported = False

            def _process_recognition(self, node, extras):
                exec_action = Exec("test_ref")
                exec_action._execute(extras)



        grammar = Grammar("test_grammar")
        grammar.add_rule(TestRule())
        grammar.load()

        try:
            grammar.process_recognition("test action1 action2")  # Simulate recognition
            self.assertTrue(action1.executed)
            self.assertTrue(action2.executed)

        finally:
            grammar.unload()

        # Test with optional and missing commands_ref
        class TestOptionalRule(CompoundRule):
            spec = "test <test_ref>"
            extras = [
                CommandsRef("test_ref", max=2, min=0), # Optional now
                RuleRef(rule=action1, name="action1"),
                RuleRef(rule=action2, name="action2"),
            ]
            exported = False

            def _process_recognition(self, node, extras):
                exec_action = Exec("test_ref")
                exec_action._execute(extras)


        grammar = Grammar("test_grammar")
        grammar.add_rule(TestOptionalRule())
        grammar.load()

        try:
            grammar.process_recognition("test") # No actions specified. Should not fail.
            self.assertFalse(action1.executed)  # Should not have executed
            self.assertFalse(action2.executed)  # Should not have executed
        finally:
            grammar.unload()



if __name__ == '__main__':
    unittest.main()