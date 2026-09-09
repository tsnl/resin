use super::*;
use resin_common::types::Conv;
use resin_common::types::check::ExplicitConversion;

impl Generator {
    pub(super) fn ascription(
        &mut self,
        span: Span,
        to: &Ty,
        source: &typed::Term,
    ) -> Result<TermKind> {
        if let Ty::Arc { pointee } = to {
            return Ok(TermKind::ArcNew {
                value: Box::new(self.shared_payload(pointee, source)?),
            });
        }
        if let Ty::Weak { pointee } = to
            && matches!(source.kind, typed::TermKind::Unit)
        {
            return Ok(TermKind::WeakEmpty {
                pointee: *pointee.clone(),
            });
        }
        let value = self.constructor_argument(to, source)?;
        self.conversion(span, value, to)
    }

    fn constructor_argument(&mut self, to: &Ty, source: &typed::Term) -> Result<Term> {
        let body = self
            .typer
            .body(to)
            .map_err(|e| GenerateError::typing(source.span, e))?;
        let body = body.span_record().unwrap_or(body);
        if matches!(&body, Ty::Record { fields } if fields.is_empty())
            && matches!(source.kind, typed::TermKind::Unit)
        {
            return Ok(Term {
                span: source.span,
                ty: body,
                kind: TermKind::Record { fields: vec![] },
            });
        }
        self.elaborate(source)
    }

    fn shared_payload(&mut self, to: &Ty, source: &typed::Term) -> Result<Term> {
        if !matches!(
            source.kind,
            typed::TermKind::Record { .. } | typed::TermKind::Unit
        ) {
            return self.elaborate(source);
        }
        let value = self.constructor_argument(to, source)?;
        if &value.ty == to {
            return Ok(value);
        }
        let kind = self.conversion(source.span, value, to)?;
        Ok(Term {
            span: source.span,
            ty: to.clone(),
            kind,
        })
    }

    fn conversion(&self, span: Span, value: Term, to: &Ty) -> Result<TermKind> {
        let conversion = self
            .typer
            .explicit_conversion(&value.ty, to)
            .map_err(|e| GenerateError::typing(span, e))?;
        if let ExplicitConversion::Ascribe(steps) = &conversion {
            self.check_unwrap(span, steps)?;
        }
        Ok(TermKind::Convert {
            conversion,
            arg: Box::new(value),
        })
    }

    fn check_unwrap(&self, span: Span, steps: &[Conv]) -> Result<()> {
        if steps.iter().any(|step| {
            matches!(step, Conv::Unwrap { definition }
            if self.typer.definition(*definition).unwrap().drop_hook().is_some())
        }) {
            return Err(GenerateError::inference(
                span,
                "cannot unwrap a type with drop; access its fields through a pointer or use Ptr.replace",
            ));
        }
        Ok(())
    }
}
