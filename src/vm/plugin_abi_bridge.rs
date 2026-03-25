//! Bridge `datacode_abi::AbiValue` ↔ [`crate::vm_value::Value`] for the ML dylib (no `data-code`).

use std::ffi::{CStr, CString, c_void};
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;

use datacode_abi::AbiValue;
use crate::ml_types::MlValueKind;
use crate::vm_value::Value;

#[derive(Debug)]
#[allow(dead_code)]
pub enum BridgeError {
    Unrepresentable(&'static str),
    InvalidUtf8,
    InvalidHandle,
}

pub struct PluginAbiBridgeContext {
    /// Keeps `Object` handles alive for `abi_to_value` lookup after argument conversion.
    object_refs: Vec<Rc<RefCell<HashMap<String, Value>>>>,
}

impl PluginAbiBridgeContext {
    pub fn new() -> Self {
        Self {
            object_refs: Vec::new(),
        }
    }

    pub fn value_to_abi(&mut self, v: &Value) -> Result<AbiValue, BridgeError> {
        match v {
            Value::Number(n) => {
                if n.fract() == 0.0 && *n >= (i64::MIN as f64) && *n <= (i64::MAX as f64) {
                    Ok(AbiValue::Int(*n as i64))
                } else {
                    Ok(AbiValue::Float(*n))
                }
            }
            Value::Bool(b) => Ok(AbiValue::Bool(*b)),
            Value::String(s) => {
                // Return a pointer that must outlive this function: the VM reads it after the dylib
                // trampoline returns, so we cannot store CString only in `self` (dropped at return).
                // Leak the allocation; this matches how C-ABI string returns are typically handled.
                let cstr = CString::new(s.as_str()).map_err(|_| BridgeError::InvalidUtf8)?;
                let ptr = cstr.as_ptr();
                std::mem::forget(cstr);
                Ok(AbiValue::Str(ptr))
            }
            Value::Null => Ok(AbiValue::Null),
            Value::Tensor(id) => Ok(AbiValue::PluginOpaque {
                tag: MlValueKind::Tensor as u8,
                id: *id,
            }),
            Value::Array(rc) => {
                let arr = rc.borrow();
                let mut abi_elems = Vec::with_capacity(arr.len());
                for elem in arr.iter() {
                    abi_elems.push(self.value_to_abi(elem)?);
                }
                // Same lifetime issue as strings: VM reads the buffer after this returns.
                let len = abi_elems.len();
                let ptr = abi_elems.as_mut_ptr();
                std::mem::forget(abi_elems);
                Ok(AbiValue::Array(ptr, len))
            }
            Value::Object(rc) => {
                self.object_refs.push(Rc::clone(rc));
                let ptr = Rc::as_ptr(self.object_refs.last().unwrap()) as *mut c_void;
                Ok(AbiValue::Object(ptr))
            }
            Value::PluginOpaque { tag, id } => Ok(AbiValue::PluginOpaque {
                tag: *tag,
                id: *id,
            }),
            Value::ObjectPtr(_) => Err(BridgeError::Unrepresentable(
                "ObjectPtr cannot be converted to ABI",
            )),
            Value::Path(_) => Err(BridgeError::Unrepresentable(
                "Path is not representable in ABI",
            )),
        }
    }

    pub fn abi_to_value(&self, a: AbiValue) -> Result<Value, BridgeError> {
        match a {
            AbiValue::Int(i) => Ok(Value::Number(i as f64)),
            AbiValue::Float(f) => Ok(Value::Number(f)),
            AbiValue::Bool(b) => Ok(Value::Bool(b)),
            AbiValue::Str(p) => {
                if p.is_null() {
                    Ok(Value::String(String::new()))
                } else {
                    let s = unsafe { CStr::from_ptr(p) }
                        .to_str()
                        .map_err(|_| BridgeError::InvalidUtf8)?;
                    Ok(Value::String(s.to_string()))
                }
            }
            AbiValue::Null => Ok(Value::Null),
            AbiValue::Array(ptr, len) => {
                if ptr.is_null() && len == 0 {
                    return Ok(Value::Array(Rc::new(RefCell::new(Vec::new()))));
                }
                if ptr.is_null() {
                    return Err(BridgeError::InvalidHandle);
                }
                let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                let mut inner = Vec::with_capacity(len);
                for &av in slice {
                    inner.push(self.abi_to_value(av)?);
                }
                Ok(Value::Array(Rc::new(RefCell::new(inner))))
            }
            AbiValue::Object(handle) => {
                if handle.is_null() {
                    return Err(BridgeError::InvalidHandle);
                }
                let ptr = handle as *const c_void;
                for rc in &self.object_refs {
                    if Rc::as_ptr(rc) as *const c_void == ptr {
                        return Ok(Value::Object(Rc::clone(rc)));
                    }
                }
                Ok(Value::ObjectPtr(handle))
            }
            AbiValue::PluginOpaque { tag, id } => {
                if tag == MlValueKind::Tensor as u8 {
                    Ok(Value::Tensor(id))
                } else {
                    Ok(Value::PluginOpaque { tag, id })
                }
            }
            AbiValue::Table {
                headers,
                headers_len,
                cells,
                rows,
                cols,
            } => {
                if rows == 0 || cols == 0 {
                    return Ok(Value::Array(Rc::new(RefCell::new(vec![
                        Value::Array(Rc::new(RefCell::new(Vec::new()))),
                        Value::Array(Rc::new(RefCell::new(Vec::new()))),
                    ]))));
                }
                if headers.is_null() || cells.is_null() {
                    return Err(BridgeError::InvalidHandle);
                }
                let expected_cells = rows
                    .checked_mul(cols)
                    .ok_or(BridgeError::InvalidHandle)?;
                let header_slice =
                    unsafe { std::slice::from_raw_parts(headers, headers_len) };
                let mut header_values = Vec::with_capacity(headers_len);
                for &av in header_slice {
                    header_values.push(self.abi_to_value(av)?);
                }
                let cells_slice =
                    unsafe { std::slice::from_raw_parts(cells, expected_cells) };
                let mut rows_vec = Vec::with_capacity(rows);
                for r in 0..rows {
                    let mut row = Vec::with_capacity(cols);
                    for c in 0..cols {
                        let av = cells_slice[r * cols + c];
                        row.push(self.abi_to_value(av)?);
                    }
                    rows_vec.push(Value::Array(Rc::new(RefCell::new(row))));
                }
                Ok(Value::Array(Rc::new(RefCell::new(vec![
                    Value::Array(Rc::new(RefCell::new(header_values))),
                    Value::Array(Rc::new(RefCell::new(rows_vec))),
                ]))))
            }
        }
    }
}

impl Default for PluginAbiBridgeContext {
    fn default() -> Self {
        Self::new()
    }
}
