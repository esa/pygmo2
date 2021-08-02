// Copyright 2020, 2021 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <iterator>
#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

#include <boost/iterator/transform_iterator.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <pagmo/problem.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/types.hpp>

#include "common_utils.hpp"
#include "problem.hpp"
#include "s11n_wrappers.hpp"

namespace pagmo
{

namespace detail
{

namespace py = pybind11;

prob_inner<py::object>::prob_inner(const py::object &o)
{
    // Forbid the use of a pygmo.problem as a UDP.
    // The motivation here is consistency with C++. In C++, the use of
    // a pagmo::problem as a UDP is forbidden and prevented by the fact
    // that the generic constructor from UDP is disabled if the input
    // object is a pagmo::problem (the copy/move constructor is
    // invoked instead). In order to achieve an equivalent behaviour
    // in pygmo, we throw an error if o is a problem, and instruct
    // the user to employ the standard copy/deepcopy facilities
    // for creating a copy of the input problem.
    if (pygmo::type(o).equal(py::module::import("pygmo").attr("problem"))) {
        pygmo::py_throw(PyExc_TypeError,
                        ("a pygmo.problem cannot be used as a UDP for another pygmo.problem (if you need to copy a "
                         "problem please use the standard Python copy()/deepcopy() functions)"));
    }
    // Check that o is an instance of a class, and not a type.
    check_not_type(o, "problem");
    // Check the presence of the mandatory methods (these are static asserts
    // in the C++ counterpart).
    check_mandatory_method(o, "fitness", "problem");
    check_mandatory_method(o, "get_bounds", "problem");
    // The Python UDP looks alright, let's deepcopy it into m_value.
    m_value = pygmo::deepcopy(o);
}

std::unique_ptr<prob_inner_base> prob_inner<py::object>::clone() const
{
    // This will make a deep copy using the ctor above.
    return std::make_unique<prob_inner>(m_value);
}

vector_double prob_inner<py::object>::fitness(const vector_double &dv) const
{
    return pygmo::ndarr_to_vector<vector_double>(
        py::cast<py::array_t<double>>(m_value.attr("fitness")(pygmo::vector_to_ndarr<py::array_t<double>>(dv))));
}

std::pair<vector_double, vector_double> prob_inner<py::object>::get_bounds() const
{
    auto tup = py::cast<py::tuple>(m_value.attr("get_bounds")());
    // Check the tuple size.
    if (py::len(tup) != 2) {
        pygmo::py_throw(PyExc_ValueError, ("the bounds of the problem must be returned as a tuple of 2 elements, but "
                                           "the detected tuple size is "
                                           + std::to_string(py::len(tup)))
                                              .c_str());
    }

    // Finally, we build the pair from the tuple elements.
    return std::make_pair(pygmo::ndarr_to_vector<vector_double>(py::cast<py::array_t<double>>(tup[0])),
                          pygmo::ndarr_to_vector<vector_double>(py::cast<py::array_t<double>>(tup[1])));
}

vector_double prob_inner<py::object>::batch_fitness(const vector_double &dv) const
{
    auto bf = pygmo::callable_attribute(m_value, "batch_fitness");
    if (bf.is_none()) {
        pygmo::py_throw(PyExc_NotImplementedError,
                        ("the batch_fitness() method has been invoked, but it is not implemented "
                         "in the user-defined Python problem '"
                         + pygmo::str(m_value) + "' of type '" + pygmo::str(pygmo::type(m_value))
                         + "': the method is either not present or not callable")
                            .c_str());
    }
    return pygmo::ndarr_to_vector<vector_double>(
        py::cast<py::array_t<double>>(bf(pygmo::vector_to_ndarr<py::array_t<double>>(dv))));
}

bool prob_inner<py::object>::has_batch_fitness() const
{
    // Same logic as in C++:
    // - without a batch_fitness() method, return false;
    // - with a batch_fitness() and no override, return true;
    // - with a batch_fitness() and override, return the value from the override.
    auto bf = pygmo::callable_attribute(m_value, "batch_fitness");
    if (bf.is_none()) {
        return false;
    }
    auto hbf = pygmo::callable_attribute(m_value, "has_batch_fitness");
    if (hbf.is_none()) {
        return true;
    }
    return py::cast<bool>(hbf());
}

vector_double::size_type prob_inner<py::object>::get_nobj() const
{
    return getter_wrapper<vector_double::size_type>(m_value, "get_nobj", 1u);
}

vector_double::size_type prob_inner<py::object>::get_nec() const
{
    return getter_wrapper<vector_double::size_type>(m_value, "get_nec", 0u);
}

vector_double::size_type prob_inner<py::object>::get_nic() const
{
    return getter_wrapper<vector_double::size_type>(m_value, "get_nic", 0u);
}

vector_double::size_type prob_inner<py::object>::get_nix() const
{
    return getter_wrapper<vector_double::size_type>(m_value, "get_nix", 0u);
}

std::string prob_inner<py::object>::get_name() const
{
    return getter_wrapper<std::string>(m_value, "get_name", pygmo::str(pygmo::type(m_value)));
}

std::string prob_inner<py::object>::get_extra_info() const
{
    return getter_wrapper<std::string>(m_value, "get_extra_info", std::string{});
}

bool prob_inner<py::object>::has_gradient() const
{
    // Same logic as in C++:
    // - without a gradient() method, return false;
    // - with a gradient() and no override, return true;
    // - with a gradient() and override, return the value from the override.
    auto g = pygmo::callable_attribute(m_value, "gradient");
    if (g.is_none()) {
        return false;
    }
    auto hg = pygmo::callable_attribute(m_value, "has_gradient");
    if (hg.is_none()) {
        return true;
    }
    return py::cast<bool>(hg());
}

vector_double prob_inner<py::object>::gradient(const vector_double &dv) const
{
    auto g = pygmo::callable_attribute(m_value, "gradient");
    if (g.is_none()) {
        pygmo::py_throw(PyExc_NotImplementedError,
                        ("gradients have been requested but they are not implemented "
                         "in the user-defined Python problem '"
                         + pygmo::str(m_value) + "' of type '" + pygmo::str(pygmo::type(m_value))
                         + "': the method is either not present or not callable")
                            .c_str());
    }
    return pygmo::ndarr_to_vector<vector_double>(
        py::cast<py::array_t<double>>(g(pygmo::vector_to_ndarr<py::array_t<double>>(dv))));
}

bool prob_inner<py::object>::has_gradient_sparsity() const
{
    // Same logic as in C++:
    // - without a gradient_sparsity() method, return false;
    // - with a gradient_sparsity() and no override, return true;
    // - with a gradient_sparsity() and override, return the value from the override.
    auto gs = pygmo::callable_attribute(m_value, "gradient_sparsity");
    if (gs.is_none()) {
        return false;
    }
    auto hgs = pygmo::callable_attribute(m_value, "has_gradient_sparsity");
    if (hgs.is_none()) {
        return true;
    }
    return py::cast<bool>(hgs());
}

sparsity_pattern prob_inner<py::object>::gradient_sparsity() const
{
    auto gs = pygmo::callable_attribute(m_value, "gradient_sparsity");
    if (gs.is_none()) {
        // NOTE: this is similar to C++: this virtual method gradient_sparsity() we are in, is called
        // only if the availability of gradient_sparsity() in the UDP was detected upon the construction
        // of a problem (i.e., m_has_gradient_sparsity is set to true). If the UDP didn't have a gradient_sparsity()
        // method upon problem construction, the m_has_gradient_sparsity is set to false and we never get here.
        // However, in Python we could have a situation in which a method is erased at runtime, so it is
        // still possible to end up in this point (if gradient_sparsity() in the internal UDP was erased
        // after the problem construction). This is something we need to strongly discourage, hence the message.
        pygmo::py_throw(PyExc_RuntimeError,
                        ("gradient sparsity has been requested but it is not implemented."
                         "This indicates a logical error in the implementation of the user-defined Python problem "
                         + pygmo::str(m_value) + "' of type '" + pygmo::str(pygmo::type(m_value))
                         + "': the gradient sparsity was available at problem construction but it has been removed "
                           "at a later stage")
                            .c_str());
    }
    return pygmo::ndarr_to_sp(py::cast<py::array_t<vector_double::size_type>>(gs()));
}

bool prob_inner<py::object>::has_hessians() const
{
    // Same logic as in C++:
    // - without a hessians() method, return false;
    // - with a hessians() and no override, return true;
    // - with a hessians() and override, return the value from the override.
    auto h = pygmo::callable_attribute(m_value, "hessians");
    if (h.is_none()) {
        return false;
    }
    auto hh = pygmo::callable_attribute(m_value, "has_hessians");
    if (hh.is_none()) {
        return true;
    }
    return py::cast<bool>(hh());
}

std::vector<vector_double> prob_inner<py::object>::hessians(const vector_double &dv) const
{
    auto h = pygmo::callable_attribute(m_value, "hessians");
    if (h.is_none()) {
        pygmo::py_throw(PyExc_NotImplementedError,
                        ("hessians have been requested but they are not implemented "
                         "in the user-defined Python problem '"
                         + pygmo::str(m_value) + "' of type '" + pygmo::str(pygmo::type(m_value))
                         + "': the method is either not present or not callable")
                            .c_str());
    }
    // Invoke the method, getting out a generic Python object.
    auto tmp = h(pygmo::vector_to_ndarr<py::array_t<double>>(dv));

    // Let's build the return value.
    struct converter {
        vector_double operator()(const py::handle &h) const
        {
            return pygmo::ndarr_to_vector<vector_double>(py::cast<py::array_t<double>>(h));
        }
    };
    auto b = boost::make_transform_iterator(std::begin(tmp), converter{});
    auto e = boost::make_transform_iterator(std::end(tmp), converter{});
    return std::vector<vector_double>(b, e);
}

bool prob_inner<py::object>::has_hessians_sparsity() const
{
    // Same logic as in C++:
    // - without a hessians_sparsity() method, return false;
    // - with a hessians_sparsity() and no override, return true;
    // - with a hessians_sparsity() and override, return the value from the override.
    auto hs = pygmo::callable_attribute(m_value, "hessians_sparsity");
    if (hs.is_none()) {
        return false;
    }
    auto hhs = pygmo::callable_attribute(m_value, "has_hessians_sparsity");
    if (hhs.is_none()) {
        return true;
    }
    return py::cast<bool>(hhs());
}

std::vector<sparsity_pattern> prob_inner<py::object>::hessians_sparsity() const
{
    auto hs = pygmo::callable_attribute(m_value, "hessians_sparsity");
    if (hs.is_none()) {
        pygmo::py_throw(PyExc_RuntimeError,
                        ("hessians sparsity has been requested but it is not implemented."
                         "This indicates a logical error in the implementation of the user-defined Python problem "
                         + pygmo::str(m_value) + "' of type '" + pygmo::str(pygmo::type(m_value))
                         + "': the hessians sparsity was available at problem construction but it has been removed "
                           "at a later stage")
                            .c_str());
    }
    auto tmp = hs();

    struct converter {
        sparsity_pattern operator()(const py::handle &h) const
        {
            return pygmo::ndarr_to_sp(py::cast<py::array_t<vector_double::size_type>>(h));
        }
    };
    auto b = boost::make_transform_iterator(std::begin(tmp), converter{});
    auto e = boost::make_transform_iterator(std::end(tmp), converter{});
    return std::vector<sparsity_pattern>(b, e);
}

void prob_inner<py::object>::set_seed(unsigned n)
{
    auto ss = pygmo::callable_attribute(m_value, "set_seed");
    if (ss.is_none()) {
        pygmo::py_throw(PyExc_NotImplementedError,
                        ("set_seed() has been invoked but it is not implemented "
                         "in the user-defined Python problem '"
                         + pygmo::str(m_value) + "' of type '" + pygmo::str(pygmo::type(m_value))
                         + "': the method is either not present or not callable")
                            .c_str());
    }
    ss(n);
}

bool prob_inner<py::object>::has_set_seed() const
{
    // Same logic as in C++:
    // - without a set_seed() method, return false;
    // - with a set_seed() and no override, return true;
    // - with a set_seed() and override, return the value from the override.
    auto ss = pygmo::callable_attribute(m_value, "set_seed");
    if (ss.is_none()) {
        return false;
    }
    auto hss = pygmo::callable_attribute(m_value, "has_set_seed");
    if (hss.is_none()) {
        return true;
    }
    return py::cast<bool>(hss());
}

// Hard code no thread safety for python problems.
thread_safety prob_inner<py::object>::get_thread_safety() const
{
    return thread_safety::none;
}

std::type_index prob_inner<py::object>::get_type_index() const
{
    return std::type_index(typeid(py::object));
}

const void *prob_inner<py::object>::get_ptr() const
{
    return &m_value;
}

void *prob_inner<py::object>::get_ptr()
{
    return &m_value;
}

template <typename Archive>
void prob_inner<py::object>::save(Archive &ar, unsigned) const
{
    pygmo::inner_class_save<prob_inner_base>(ar, *this);
}

template <typename Archive>
void prob_inner<py::object>::load(Archive &ar, unsigned)
{
    pygmo::inner_class_load<prob_inner_base>(ar, *this);
}

} // namespace detail

} // namespace pagmo

PAGMO_S11N_PROBLEM_IMPLEMENT(pybind11::object)
