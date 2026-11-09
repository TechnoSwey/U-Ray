# Makefile для U-Ray

URAY := ./uray
TESTS := ./run_tests.sh

.PHONY: test fast-test examples clean help

help:
	@echo "make test       — запустить все автотесты (включая ML)"
	@echo "make fast-test  — только быстрые тесты (без ml_*)"
	@echo "make examples   — прогнать все примеры"
	@echo "make clean      — удалить сгенерированные JSON и __pycache__"

test:
	@$(TESTS)

fast-test:
	@$(TESTS) --fast

examples:
	@for f in examples/*.ur; do \
		echo "=== $$f ==="; \
		$(URAY) "$$f" > /dev/null && echo "OK"; \
	done

clean:
	@rm -f examples/*.json *.json
	@rm -rf __pycache__ examples/__pycache__ tests/__pycache__
	@echo "cleaned"
